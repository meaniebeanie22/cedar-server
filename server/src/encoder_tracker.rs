//! Encoder-based dead reckoning for Cedar.
//!
//! An `ImuTrait` implementation for an alt-az mount fitted with two quadrature
//! encoders (encoder A on the azimuth axis, encoder B on the altitude axis).
//!
//! * A background thread sleeps until a GPIO edge interrupt arrives, decodes
//!   the encoders, and records their counts whenever they change.
//! * Plate solves are matched to the counts at the time they were taken, and
//!   used to fit a model (see `pointing_model` below) that turns encoder
//!   counts into altitude/azimuth between solves.
//! * `get_state` reports the axis rotation rates as a gyro, so Cedar can see
//!   how fast the mount is moving. There is no accelerometer.
//!
//! Dependencies: async-trait, log, nalgebra, quadrature-decoder, rppal.

use async_trait::async_trait;
use canonical_error::{CanonicalError, CanonicalErrorCode};
use cedar_elements::imu_trait::{
    AccelData, GyroData, HorizonCoordinates, ImuState, ImuTrait, TrackerState,
    TransformCalibration, ZeroBias,
};
use quadrature_decoder::{IncrementalDecoder, QuadStep};
use rppal::gpio::{Gpio, InputPin, Trigger};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::thread;
use std::time::{Duration, SystemTime};

use pointing_model::{CalibrationPoint, PointingModel};

// ---------------------------------------------------------------------------
// Constants. Everything you might want to tune lives up here.
// ---------------------------------------------------------------------------

/// BCM pin numbers: (encoder A clk, encoder A dt, encoder B clk, encoder B dt).
/// PLACEHOLDERS: BCM 0/1 are reserved and 2/3 are the I2C pins; pick free ones.
const ENCODER_PINS: (u8, u8, u8, u8) = (5, 6, 13, 19);

/// Counts per full revolution, as the decoder reports them. With `QuadStep`
/// that is 4 x the encoder's pulses per revolution (1000 P/R -> 4000). If your
/// "1000 count" is already counts after x4 decoding, set this to 1000.
const COUNTS_PER_REV: i64 = 4000;

/// The interrupt wait gives up after this long so the thread can notice that
/// the tracker was dropped. It does not affect how fast edges are handled.
const WAKE_INTERVAL: Duration = Duration::from_millis(100);

/// How far back the movement history reaches. Must comfortably exceed the
/// delay between an exposure and its plate solve being reported.
const MOVEMENT_HISTORY: Duration = Duration::from_secs(10);
/// Hard cap on the history length, as a memory safety net.
const MOVEMENT_MAX_LEN: usize = 50_000;

/// The reported rotation rate is averaged over this long. A shorter window
/// reacts faster but is coarser: one count is 360/COUNTS_PER_REV degrees, so
/// the slowest non-zero rate is that divided by this many seconds.
const GYRO_WINDOW: Duration = Duration::from_millis(200);

/// No count change for this long means the mount has stopped.
const MOVING_TIMEOUT: Duration = Duration::from_millis(20);

const REGRESSION_DATA_LEN: usize = 50;
/// A fit worse than this (RMS, degrees) is logged as a warning.
const WARN_FIT_RMS_DEG: f64 = 1.0;
const MIN_SOLVES_FOR_REGRESSION: usize = 3;
/// A new solve is ignored unless an encoder moved at least this many counts
/// since the last stored solve (0.5% of a revolution).
const MIN_COUNT_CHANGE: i64 = COUNTS_PER_REV / 200;

// ---------------------------------------------------------------------------
// Data types.
// ---------------------------------------------------------------------------

type Decoder = IncrementalDecoder<QuadStep, i32>;

/// Encoder counts at one instant. Only recorded when the counts CHANGE.
#[derive(Clone, Copy)]
struct MovementDatapoint {
    time: SystemTime,
    a: i64,
    b: i64,
}

/// Everything the interrupt thread writes, grouped so ONE mutex protects it.
#[derive(Default)]
struct MovementState {
    /// Front is newest, back is oldest.
    history: VecDeque<MovementDatapoint>,
    /// When the counts last changed. `None` if they never have.
    last_change: Option<SystemTime>,
}

impl MovementState {
    /// Adds a new datapoint and drops history that is no longer needed.
    fn push(&mut self, datapoint: MovementDatapoint) {
        self.history.push_front(datapoint);

        // Drop entries older than the window, but always keep the newest entry
        // that is older than the cutoff: it records where the encoders sat
        // *during* the start of the window, which a lookup may still need.
        if let Some(cutoff) = datapoint.time.checked_sub(MOVEMENT_HISTORY) {
            while self.history.len() > 1
                && self.history[self.history.len() - 2].time < cutoff
            {
                self.history.pop_back();
            }
        }
        while self.history.len() > MOVEMENT_MAX_LEN {
            self.history.pop_back();
        }
    }

    /// Average encoder speed over the last `window`, as (A, B) in counts per
    /// second. `None` until there is any history. Because only changes are
    /// recorded, a mount that has been still for longer than `window` reads as
    /// exactly zero.
    fn rate_counts_per_sec(
        &self,
        now: SystemTime,
        window: Duration,
    ) -> Option<(f64, f64)> {
        let newest = self.history.front()?;
        let oldest = self.history.back()?;
        let start = now.checked_sub(window)?;

        // Just after startup the history may not reach back a full window, so
        // measure over what exists.
        let ((a0, b0), seconds) = match self.counts_at(&start) {
            Some(counts) => (counts, window.as_secs_f64()),
            None => (
                (oldest.a, oldest.b),
                now.duration_since(oldest.time)
                    .ok()?
                    .as_secs_f64()
                    .max(0.001),
            ),
        };
        Some((
            (newest.a - a0) as f64 / seconds,
            (newest.b - b0) as f64 / seconds,
        ))
    }

    /// Encoder counts as of `time`: the newest entry at or before it. Because
    /// we only record changes, that entry is still correct if it is old.
    fn counts_at(&self, time: &SystemTime) -> Option<(i64, i64)> {
        self.history
            .iter()
            .find(|datapoint| datapoint.time <= *time)
            .map(|datapoint| (datapoint.a, datapoint.b))
    }
}

#[derive(Clone, Copy)]
struct PlatesolveDatapoint {
    platesolve_result: HorizonCoordinates,
    encoder_a_count: i64, // wrapped into 0..COUNTS_PER_REV
    encoder_b_count: i64,
}

// ---------------------------------------------------------------------------
// Small helpers.
// ---------------------------------------------------------------------------

fn failed_precondition(message: impl Into<String>) -> CanonicalError {
    CanonicalError {
        code: CanonicalErrorCode::FailedPrecondition,
        message: message.into(),
    }
}

/// Turns a poisoned-mutex error into a `CanonicalError`, for use with `?`.
fn poisoned<T>(error: PoisonError<T>) -> CanonicalError {
    failed_precondition(error.to_string())
}

fn push_bounded<T>(buffer: &mut VecDeque<T>, max_len: usize, element: T) {
    buffer.push_front(element);
    if buffer.len() > max_len {
        buffer.pop_back();
    }
}

/// Encoder counts to degrees of rotation.
fn counts_to_deg(counts: i64) -> f64 {
    counts as f64 * 360.0 / COUNTS_PER_REV as f64
}

/// Shortest distance between two wrapped counts (3999 and 1 are 2 apart).
fn circular_diff(a: i64, b: i64) -> i64 {
    let d = (a - b).rem_euclid(COUNTS_PER_REV);
    d.min(COUNTS_PER_REV - d)
}

fn make_pin(gpio: &Gpio, pin: u8) -> InputPin {
    gpio.get(pin)
        .expect("Couldn't init an encoder pin")
        .into_input_pullup()
}

/// Gives a decoder the pins' starting levels, so it knows where it begins, then
/// zeroes its count. (Its default assumption is that both pins start low.)
fn seed_decoder(decoder: &mut Decoder, a: bool, b: bool) {
    let _ = decoder.update(a, b);
    decoder.set_counter(0);
}

// ---------------------------------------------------------------------------
// The tracker struct.
// ---------------------------------------------------------------------------

pub struct EncoderTracker {
    /// Parked here by `new()`, then moved out into the interrupt thread by
    /// `start()`. After that this is `None`. Order: A clk, A dt, B clk, B dt.
    pins: Mutex<Option<(Gpio, [InputPin; 4])>>,
    /// Written by the interrupt thread, read by the trait methods.
    movement: Arc<Mutex<MovementState>>,
    /// Front is new, back is old.
    platesolve_data: Mutex<VecDeque<PlatesolveDatapoint>>,
    /// Cached fit of encoder angles to sky coordinates. Cleared whenever a
    /// new solve is stored, and refitted on demand.
    model: Mutex<Option<PointingModel>>,
    /// Tells the interrupt thread to keep going (true) or exit (false).
    running: Arc<AtomicBool>,
}

impl EncoderTracker {
    pub fn new() -> EncoderTracker {
        let gpio = Gpio::new().expect("Couldn't init a gpio");
        let pins = [
            make_pin(&gpio, ENCODER_PINS.0),
            make_pin(&gpio, ENCODER_PINS.1),
            make_pin(&gpio, ENCODER_PINS.2),
            make_pin(&gpio, ENCODER_PINS.3),
        ];

        EncoderTracker {
            pins: Mutex::new(Some((gpio, pins))),
            movement: Arc::new(Mutex::new(MovementState::default())),
            platesolve_data: Mutex::new(VecDeque::with_capacity(
                REGRESSION_DATA_LEN,
            )),
            model: Mutex::new(None),
            running: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Fits the pointing model to the stored solves and caches it. Lock order
    /// is always platesolve_data first, then model.
    fn do_platesolve_regression(
        &self,
    ) -> Result<PointingModel, CanonicalError> {
        let solves = self.platesolve_data.lock().map_err(poisoned)?;
        if solves.len() < MIN_SOLVES_FOR_REGRESSION {
            return Err(failed_precondition("Not enough platesolves yet"));
        }

        // Encoder A is the azimuth axis, encoder B the altitude axis. The
        // model copes with either being reversed, but not with them swapped.
        let points: Vec<CalibrationPoint> = solves
            .iter()
            .map(|solve| CalibrationPoint {
                encoder_az_deg: counts_to_deg(solve.encoder_a_count),
                encoder_alt_deg: counts_to_deg(solve.encoder_b_count),
                true_az_deg: solve.platesolve_result.azimuth,
                true_alt_deg: solve.platesolve_result.altitude,
                true_zenith_roll_deg: solve.platesolve_result.zenith_roll_angle,
            })
            .collect();

        let model = PointingModel::fit(&points).map_err(failed_precondition)?;
        if model.rms_error_deg > WARN_FIT_RMS_DEG {
            log::warn!(
                "Pointing model fits the platesolves poorly ({:.2} deg RMS)",
                model.rms_error_deg
            );
        }

        *self.model.lock().map_err(poisoned)? = Some(model);
        Ok(model)
    }
}

/// When the tracker is dropped, tell the interrupt thread to exit.
impl Drop for EncoderTracker {
    fn drop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
    }
}

// ---------------------------------------------------------------------------
// The interrupt loop. Runs on its own thread and owns the four pins.
// ---------------------------------------------------------------------------

fn interrupt_loop(
    gpio: Gpio,
    mut pins: [InputPin; 4],
    movement: Arc<Mutex<MovementState>>,
    running: Arc<AtomicBool>,
) {
    // Ask the kernel to queue an event on every rising AND falling edge.
    for pin in pins.iter_mut() {
        pin.set_interrupt(Trigger::Both, None)
            .expect("Couldn't enable an interrupt on an encoder pin");
    }
    // From here on we only need shared (read-only) access to the pins.
    let pins = pins;
    let watched: Vec<&InputPin> = pins.iter().collect();

    let mut decoder_a = Decoder::default();
    let mut decoder_b = Decoder::default();
    seed_decoder(&mut decoder_a, pins[0].is_high(), pins[1].is_high());
    seed_decoder(&mut decoder_b, pins[2].is_high(), pins[3].is_high());
    let mut last = (0i64, 0i64);

    // Record where we started, so there is always at least one datapoint.
    movement
        .lock()
        .expect("movement mutex poisoned")
        .push(MovementDatapoint {
            time: SystemTime::now(),
            a: last.0,
            b: last.1,
        });

    // An Err from update() means the decoder saw an impossible state jump,
    // i.e. an edge was missed or arrived out of order.
    let mut bad_reads: u32 = 0;

    while running.load(Ordering::Relaxed) {
        // Sleeps (using no CPU) until any watched pin has an edge, or the
        // timeout passes. `false` = keep edges that queued up while we were busy.
        match gpio.poll_interrupts(&watched, false, Some(WAKE_INTERVAL)) {
            Ok(Some(_)) => {}
            Ok(None) => continue, // timeout: loop round and re-check `running`
            Err(error) => {
                log::error!("Waiting for encoder interrupts failed: {error}");
                thread::sleep(WAKE_INTERVAL);
                continue;
            }
        }

        // Something changed: read all four pins and feed both decoders.
        let errors = u32::from(
            decoder_a
                .update(pins[0].is_high(), pins[1].is_high())
                .is_err(),
        ) + u32::from(
            decoder_b
                .update(pins[2].is_high(), pins[3].is_high())
                .is_err(),
        );
        if errors > 0 {
            let before = bad_reads;
            bad_reads += errors;
            if before == 0 || before / 100 != bad_reads / 100 {
                log::warn!(
                    "Encoder decoding missed a step ({bad_reads} total)."
                );
            }
        }

        let now = (
            i64::from(decoder_a.counter()),
            i64::from(decoder_b.counter()),
        );
        if now != last {
            let time = SystemTime::now();
            let mut state = movement.lock().expect("movement mutex poisoned");
            state.push(MovementDatapoint {
                time,
                a: now.0,
                b: now.1,
            });
            state.last_change = Some(time);
            drop(state);
            last = now;
        }
    }
}

// ---------------------------------------------------------------------------
// The ImuTrait implementation.
// ---------------------------------------------------------------------------

#[async_trait]
impl ImuTrait for EncoderTracker {
    // Spawns the interrupt thread. Moves the pins into it.
    fn start(&self) {
        let Some((gpio, pins)) =
            self.pins.lock().expect("pins mutex poisoned").take()
        else {
            log::warn!("EncoderTracker::start() called more than once");
            return;
        };

        self.running.store(true, Ordering::Relaxed);
        let movement = Arc::clone(&self.movement);
        let running = Arc::clone(&self.running);

        thread::Builder::new()
            .name("encoder-irq".into())
            .spawn(move || interrupt_loop(gpio, pins, movement, running))
            .expect("Couldn't spawn the encoder interrupt thread");
    }

    // Looks up the encoder counts at the time of the solve.
    async fn report_true_camera_pointing(
        &self,
        camera_pointing: &HorizonCoordinates,
        timestamp: &SystemTime,
    ) {
        // The lock is released at the end of this statement.
        let counts = self
            .movement
            .lock()
            .expect("movement mutex poisoned")
            .counts_at(timestamp);

        let Some((a, b)) = counts else {
            log::warn!(
                "No encoder history covers this platesolve; ignoring it"
            );
            return;
        };
        let a = a.rem_euclid(COUNTS_PER_REV);
        let b = b.rem_euclid(COUNTS_PER_REV);

        let mut solves = self
            .platesolve_data
            .lock()
            .expect("Unable to lock platesolve_data to add new solve");

        // Ignore the solve if neither encoder moved enough since the last one.
        if let Some(latest) = solves.front() {
            if circular_diff(latest.encoder_a_count, a) < MIN_COUNT_CHANGE
                && circular_diff(latest.encoder_b_count, b) < MIN_COUNT_CHANGE
            {
                return;
            }
        }

        push_bounded(
            &mut solves,
            REGRESSION_DATA_LEN,
            PlatesolveDatapoint {
                platesolve_result: *camera_pointing,
                encoder_a_count: a,
                encoder_b_count: b,
            },
        );

        // The cached fit is out of date now. (Lock order: solves, then model.)
        *self
            .model
            .lock()
            .expect("Couldn't invalidate the pointing model") = None;
    }

    // No plate solution is available. This is either due to visual obstruction
    // (clouds, etc) or platform motion preventing star detection.
    async fn report_camera_pointing_lost(&self) {}

    // Discards all calibration state. The raw movement history is a
    // measurement, not calibration, so it stays.
    async fn reset(&self) {
        *self.model.lock().expect("Couldn't reset pointing model") = None;
        self.platesolve_data
            .lock()
            .expect("Couldn't reset platesolve_data")
            .clear();
    }

    async fn get_estimated_camera_pointing(
        &self,
        timestamp: &SystemTime,
    ) -> Result<HorizonCoordinates, CanonicalError> {
        // Where were the encoders at that time? (Lock released straight away.)
        let counts =
            self.movement.lock().map_err(poisoned)?.counts_at(timestamp);
        let Some((a, b)) = counts else {
            return Err(failed_precondition(
                "No encoder history covers the requested time",
            ));
        };

        // Copy the cached model out so the lock is released before a refit
        // (which locks again) can run.
        let cached = *self.model.lock().map_err(poisoned)?;
        let model = match cached {
            Some(model) => model,
            None => self.do_platesolve_regression()?,
        };

        let (azimuth, altitude) =
            model.predict(counts_to_deg(a), counts_to_deg(b));
        Ok(HorizonCoordinates {
            zenith_roll_angle: model.zenith_roll_deg,
            altitude,
            azimuth,
        })
    }

    // Tracker state comes from when the counts last changed.
    async fn get_tracker_state(&self) -> TrackerState {
        let Ok(state) = self.movement.lock() else {
            return TrackerState::Lost;
        };

        let Some(last_change) = state.last_change else {
            // Never moved. Lost if the thread hasn't produced a reading yet.
            return if state.history.is_empty() {
                TrackerState::Lost
            } else {
                TrackerState::Motionless
            };
        };

        match SystemTime::now().duration_since(last_change) {
            Ok(elapsed) if elapsed < MOVING_TIMEOUT => TrackerState::Moving,
            Ok(_) => TrackerState::Motionless,
            Err(_) => TrackerState::Lost,
        }
    }

    async fn get_calibration(
        &self,
    ) -> (Option<ZeroBias>, Option<TransformCalibration>) {
        (None, None)
    }

    // Reports the axis rotation rates as a gyro: x is the azimuth axis, y the
    // altitude axis (degrees/second, signed by the encoder's own direction, so
    // a reversed encoder reads negative), and z is 0 because nothing measures
    // rotation about the line of sight. There is no accelerometer, so accel
    // is all zeros.
    async fn get_state(
        &self,
    ) -> Result<(ImuState, SystemTime), CanonicalError> {
        let now = SystemTime::now();
        let rate = self
            .movement
            .lock()
            .map_err(poisoned)?
            .rate_counts_per_sec(now, GYRO_WINDOW);
        let Some((rate_a, rate_b)) = rate else {
            return Err(failed_precondition("No encoder readings yet"));
        };

        let deg_per_count = 360.0 / COUNTS_PER_REV as f64;
        Ok((
            ImuState {
                timestamp: now,
                accel: AccelData::default(),
                gyro: GyroData {
                    x: rate_a * deg_per_count,
                    y: rate_b * deg_per_count,
                    z: 0.0,
                },
            },
            now,
        ))
    }

    async fn get_angular_velocity_magnitude(
        &self,
    ) -> Result<(f64, SystemTime), CanonicalError> {
        let (state, time) = self.get_state().await?;
        Ok((state.gyro.x.hypot(state.gyro.y), time))
    }

    async fn get_jerk_magnitude(
        &self,
    ) -> Result<(f64, SystemTime), CanonicalError> {
        // Jerk comes from an accelerometer, which this tracker doesn't have.
        Err(failed_precondition("EncoderTracker has no accelerometer"))
    }

    fn get_model(&self) -> String {
        String::from("EncoderTracker")
    }

    fn save_state(&self) -> Result<(), CanonicalError> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// The pointing model: turns encoder angles into altitude/azimuth.
// ---------------------------------------------------------------------------

mod pointing_model {
    //! Maps encoder angles to horizon coordinates, calibrated from plate solves.
    //!
    //! The camera boresight is treated as a unit vector (x = north, y = east,
    //! z = up, so azimuth is measured clockwise from north). From the encoders we
    //! build
    //!
    //! ```text
    //! u = unit_vector(encoder_az, encoder_alt + alt_offset)
    //! ```
    //!
    //! and the true pointing is `v = R * u`, where `R` is a rotation.
    //!
    //! * `R` absorbs the azimuth zero offset AND any tilt of the azimuth axis.
    //! * `alt_offset` is the altitude zero offset. It cannot be folded into `R`,
    //!   because it rotates about a different axis at every azimuth, so it is
    //!   searched for separately, over a full circle.
    //! * Reversed encoders need no special handling. On a sphere, reversing the
    //!   azimuth direction is exactly the same as adding 180 degrees to the
    //!   altitude offset and rotating, and reversing altitude is the same again
    //!   combined with a reversed azimuth. So the fit above already covers all
    //!   four direction combinations.
    //!
    //! Everything here is plain maths with no hardware, so it is easy to test.

    use nalgebra::{Matrix3, Rotation3, Vector3};

    /// Fewest calibration points a fit will accept.
    const MIN_POINTS: usize = 3;
    /// At least two calibration points must be this far apart on the sky,
    /// otherwise the fit has nothing to pin the rotation down with.
    const MIN_SPREAD_DEG: f64 = 10.0;
    /// Step of the coarse search over the altitude zero offset.
    const ALT_OFFSET_GRID_STEP_DEG: f64 = 1.0;

    /// One plate solve paired with the encoder angles at that moment.
    #[derive(Clone, Copy, Debug)]
    pub struct CalibrationPoint {
        /// Encoder A (azimuth axis) in degrees, as reported.
        pub encoder_az_deg: f64,
        /// Encoder B (altitude axis) in degrees, as reported.
        pub encoder_alt_deg: f64,
        pub true_az_deg: f64,
        pub true_alt_deg: f64,
        pub true_zenith_roll_deg: f64,
    }

    #[derive(Clone, Copy, Debug)]
    pub struct PointingModel {
        alt_offset_deg: f64,
        rotation: Rotation3<f64>,
        /// RMS angular error of the fit over the calibration points, in degrees.
        pub rms_error_deg: f64,
        /// Zenith roll angle in the camera image. On an alt-az mount with the
        /// camera fixed to the tube this does not change with pointing, so it is
        /// simply the circular mean of the solves' values.
        pub zenith_roll_deg: f64,
    }

    impl PointingModel {
        /// Fits the model to the calibration points.
        pub fn fit(
            points: &[CalibrationPoint],
        ) -> Result<PointingModel, String> {
            if points.len() < MIN_POINTS {
                return Err(format!(
                    "Need at least {MIN_POINTS} calibration points, have {}",
                    points.len()
                ));
            }

            let true_vectors: Vec<Vector3<f64>> = points
                .iter()
                .map(|p| unit_vector(p.true_az_deg, p.true_alt_deg))
                .collect();

            if max_separation_deg(&true_vectors) < MIN_SPREAD_DEG {
                return Err(format!(
                    "Calibration points are all within {MIN_SPREAD_DEG} degrees of \
                     each other. Point the scope at places spread across the sky."
                ));
            }

            let (alt_offset_deg, rotation, rms_error_deg) =
                best_alt_offset(points, &true_vectors);
            let rolls: Vec<f64> =
                points.iter().map(|p| p.true_zenith_roll_deg).collect();

            Ok(PointingModel {
                alt_offset_deg,
                rotation,
                rms_error_deg,
                zenith_roll_deg: circular_mean_deg(&rolls),
            })
        }

        /// Returns (azimuth, altitude) in degrees: azimuth in 0..360.
        pub fn predict(
            &self,
            encoder_az_deg: f64,
            encoder_alt_deg: f64,
        ) -> (f64, f64) {
            let u = unit_vector(
                encoder_az_deg,
                encoder_alt_deg + self.alt_offset_deg,
            );
            let v = self.rotation * u;
            let az = v.y.atan2(v.x).to_degrees().rem_euclid(360.0);
            let alt = v.z.clamp(-1.0, 1.0).asin().to_degrees();
            (az, alt)
        }
    }

    // ---------------------------------------------------------------------------
    // Internals
    // ---------------------------------------------------------------------------

    fn unit_vector(az_deg: f64, alt_deg: f64) -> Vector3<f64> {
        let (az, alt) = (az_deg.to_radians(), alt_deg.to_radians());
        Vector3::new(alt.cos() * az.cos(), alt.cos() * az.sin(), alt.sin())
    }

    /// Angle between two vectors in radians. Stays accurate for tiny angles,
    /// unlike `acos` of the dot product.
    fn angle_between(a: &Vector3<f64>, b: &Vector3<f64>) -> f64 {
        a.cross(b).norm().atan2(a.dot(b))
    }

    fn max_separation_deg(vectors: &[Vector3<f64>]) -> f64 {
        let mut max = 0.0f64;
        for (i, a) in vectors.iter().enumerate() {
            for b in &vectors[i + 1..] {
                max = max.max(angle_between(a, b).to_degrees());
            }
        }
        max
    }

    fn wrap180(d: f64) -> f64 {
        (d + 180.0).rem_euclid(360.0) - 180.0
    }

    /// Mean of angles in degrees that copes with wraparound (359 and 1 -> 0).
    fn circular_mean_deg(values: &[f64]) -> f64 {
        let (mut sin_sum, mut cos_sum) = (0.0, 0.0);
        for v in values {
            sin_sum += v.to_radians().sin();
            cos_sum += v.to_radians().cos();
        }
        sin_sum.atan2(cos_sum).to_degrees()
    }

    /// The rotation R minimising the sum of squared angles between `R * u_i` and
    /// `v_i` (the Wahba problem, solved by SVD, a.k.a. the Kabsch algorithm).
    fn kabsch(us: &[Vector3<f64>], vs: &[Vector3<f64>]) -> Rotation3<f64> {
        let mut h = Matrix3::<f64>::zeros();
        for (u, v) in us.iter().zip(vs) {
            h += u * v.transpose();
        }
        let svd = h.svd(true, true);
        let u = svd.u.expect("SVD was asked for U");
        let v_t = svd.v_t.expect("SVD was asked for V^T");
        // Without this, the answer can be a mirror image instead of a rotation.
        let d = (v_t.transpose() * u.transpose()).determinant().signum();
        let r = v_t.transpose()
            * Matrix3::from_diagonal(&Vector3::new(1.0, 1.0, d))
            * u.transpose();
        Rotation3::from_matrix_unchecked(r)
    }

    /// Best rotation for one altitude offset, plus its mean squared angle
    /// (radians^2).
    fn evaluate(
        points: &[CalibrationPoint],
        true_vectors: &[Vector3<f64>],
        alt_offset_deg: f64,
    ) -> (Rotation3<f64>, f64) {
        let us: Vec<Vector3<f64>> = points
            .iter()
            .map(|p| {
                unit_vector(
                    p.encoder_az_deg,
                    p.encoder_alt_deg + alt_offset_deg,
                )
            })
            .collect();
        let rotation = kabsch(&us, true_vectors);
        let sum_sq: f64 = us
            .iter()
            .zip(true_vectors)
            .map(|(u, v)| angle_between(&(rotation * u), v).powi(2))
            .sum();
        (rotation, sum_sq / points.len() as f64)
    }

    /// Finds the altitude offset (and matching rotation) that fits best: a coarse
    /// sweep over a full circle, then a golden-section search around the winner.
    /// Returns (offset in degrees, rotation, RMS error in degrees).
    fn best_alt_offset(
        points: &[CalibrationPoint],
        true_vectors: &[Vector3<f64>],
    ) -> (f64, Rotation3<f64>, f64) {
        let cost = |offset: f64| evaluate(points, true_vectors, offset).1;

        let mut best_offset = -180.0;
        let mut best_cost = f64::MAX;
        let mut offset = -180.0;
        while offset < 180.0 {
            let c = cost(offset);
            if c < best_cost {
                best_cost = c;
                best_offset = offset;
            }
            offset += ALT_OFFSET_GRID_STEP_DEG;
        }

        let golden = (5.0_f64.sqrt() - 1.0) / 2.0;
        let mut lo = best_offset - ALT_OFFSET_GRID_STEP_DEG;
        let mut hi = best_offset + ALT_OFFSET_GRID_STEP_DEG;
        for _ in 0..40 {
            let x1 = hi - golden * (hi - lo);
            let x2 = lo + golden * (hi - lo);
            if cost(x1) < cost(x2) {
                hi = x2;
            } else {
                lo = x1;
            }
        }

        let alt_offset_deg = wrap180(0.5 * (lo + hi));
        let (rotation, mean_sq) =
            evaluate(points, true_vectors, alt_offset_deg);
        (alt_offset_deg, rotation, mean_sq.sqrt().to_degrees())
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// A made-up mount: encoder directions, altitude zero offset, and a
        /// rotation covering azimuth offset + axis tilt.
        struct Truth {
            az_sign: f64,
            alt_sign: f64,
            alt_offset_deg: f64,
            rotation: Rotation3<f64>,
        }

        impl Truth {
            fn pointing(&self, enc_az: f64, enc_alt: f64) -> (f64, f64) {
                let u = unit_vector(
                    self.az_sign * enc_az,
                    self.alt_sign * enc_alt + self.alt_offset_deg,
                );
                let v = self.rotation * u;
                (
                    v.y.atan2(v.x).to_degrees().rem_euclid(360.0),
                    v.z.asin().to_degrees(),
                )
            }
        }

        const ENCODER_SPREAD: [(f64, f64); 6] = [
            (10.0, 20.0),
            (95.0, 45.0),
            (190.0, 60.0),
            (260.0, 30.0),
            (350.0, 75.0),
            (120.0, 10.0),
        ];

        fn points_for(truth: &Truth, noise_deg: f64) -> Vec<CalibrationPoint> {
            // Tiny deterministic pseudo-noise, so the tests need no extra crate.
            let mut state: u32 = 12345;
            let mut noise = move || {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                ((state >> 8) as f64 / (1u32 << 24) as f64 - 0.5)
                    * 2.0
                    * noise_deg
            };
            ENCODER_SPREAD
                .iter()
                .map(|&(a, b)| {
                    let (az, alt) = truth.pointing(a, b);
                    CalibrationPoint {
                        encoder_az_deg: a,
                        encoder_alt_deg: b,
                        true_az_deg: az + noise(),
                        true_alt_deg: alt + noise(),
                        true_zenith_roll_deg: 0.0,
                    }
                })
                .collect()
        }

        fn assert_predicts(model: &PointingModel, truth: &Truth, tol_deg: f64) {
            for (a, b) in [(200.0, 40.0), (300.0, 65.0), (45.0, 15.0)] {
                let (want_az, want_alt) = truth.pointing(a, b);
                let (az, alt) = model.predict(a, b);
                assert!(
                    wrap180(az - want_az).abs() < tol_deg,
                    "az {az} vs {want_az}"
                );
                assert!(
                    (alt - want_alt).abs() < tol_deg,
                    "alt {alt} vs {want_alt}"
                );
            }
        }

        #[test]
        fn recovers_offsets_tilt_and_every_encoder_direction() {
            for az_sign in [1.0, -1.0] {
                for alt_sign in [1.0, -1.0] {
                    let truth = Truth {
                        az_sign,
                        alt_sign,
                        alt_offset_deg: 33.0,
                        rotation: Rotation3::from_euler_angles(
                            0.02, -0.015, 1.1,
                        ),
                    };
                    let model =
                        PointingModel::fit(&points_for(&truth, 0.0)).unwrap();
                    assert!(
                        model.rms_error_deg < 1e-3,
                        "{}",
                        model.rms_error_deg
                    );
                    assert_predicts(&model, &truth, 1e-3);
                }
            }
        }

        #[test]
        fn recovers_an_altitude_zero_offset_on_its_own() {
            // The case a plain rotation fit cannot handle.
            let truth = Truth {
                az_sign: 1.0,
                alt_sign: 1.0,
                alt_offset_deg: 17.0,
                rotation: Rotation3::identity(),
            };
            let model = PointingModel::fit(&points_for(&truth, 0.0)).unwrap();
            assert!(model.rms_error_deg < 1e-3);
            assert_predicts(&model, &truth, 1e-3);
        }

        #[test]
        fn tolerates_small_solve_noise() {
            let truth = Truth {
                az_sign: -1.0,
                alt_sign: 1.0,
                alt_offset_deg: -120.0,
                rotation: Rotation3::from_euler_angles(0.01, 0.02, -2.0),
            };
            let model = PointingModel::fit(&points_for(&truth, 0.05)).unwrap();
            assert!(model.rms_error_deg < 0.1, "{}", model.rms_error_deg);
            assert_predicts(&model, &truth, 0.2);
        }

        #[test]
        fn needs_enough_points() {
            let truth = Truth {
                az_sign: 1.0,
                alt_sign: 1.0,
                alt_offset_deg: 0.0,
                rotation: Rotation3::identity(),
            };
            let points = points_for(&truth, 0.0);
            assert!(PointingModel::fit(&points[..2]).is_err());
        }

        #[test]
        fn refuses_points_bunched_together() {
            let same = CalibrationPoint {
                encoder_az_deg: 50.0,
                encoder_alt_deg: 30.0,
                true_az_deg: 80.0,
                true_alt_deg: 35.0,
                true_zenith_roll_deg: 0.0,
            };
            assert!(PointingModel::fit(&[same, same, same]).is_err());
        }

        #[test]
        fn zenith_roll_is_a_circular_mean() {
            let truth = Truth {
                az_sign: 1.0,
                alt_sign: 1.0,
                alt_offset_deg: 5.0,
                rotation: Rotation3::from_euler_angles(0.0, 0.0, 0.5),
            };
            let mut points = points_for(&truth, 0.0);
            for (i, p) in points.iter_mut().enumerate() {
                p.true_zenith_roll_deg = if i % 2 == 0 { 359.0 } else { 1.0 };
            }
            let model = PointingModel::fit(&points).unwrap();
            assert!(wrap180(model.zenith_roll_deg).abs() < 1e-6);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn datapoint(base: SystemTime, ms: u64, a: i64) -> MovementDatapoint {
        MovementDatapoint {
            time: base + Duration::from_millis(ms),
            a,
            b: 0,
        }
    }

    #[test]
    fn counts_at_finds_newest_entry_not_after_the_time() {
        let base = SystemTime::UNIX_EPOCH + Duration::from_secs(1_000);
        let mut state = MovementState::default();
        state.push(datapoint(base, 0, 10));
        state.push(datapoint(base, 100, 20));
        state.push(datapoint(base, 200, 30));

        let at = |ms| state.counts_at(&(base + Duration::from_millis(ms)));
        assert_eq!(at(150), Some((20, 0)));
        assert_eq!(at(500), Some((30, 0)));
        assert_eq!(state.counts_at(&(base - Duration::from_millis(1))), None);
    }

    #[test]
    fn pruning_keeps_the_entry_that_covers_the_start_of_the_window() {
        let base = SystemTime::UNIX_EPOCH + Duration::from_secs(1_000);
        let mut state = MovementState::default();
        state.push(datapoint(base, 0, 10));
        state.push(datapoint(base, 30_000, 20));
        state.push(datapoint(base, 31_000, 25));
        state.push(datapoint(base, 45_000, 30));

        // Window is 35 s..45 s, so the 31 s entry must survive: it is where
        // the encoders sat at 35 s.
        assert_eq!(state.history.len(), 2);
        let at_40s = state.counts_at(&(base + Duration::from_millis(40_000)));
        assert_eq!(at_40s, Some((25, 0)));
    }

    /// Walks the decoder through full quadrature cycles, one edge at a time.
    fn walk(decoder: &mut Decoder, states: &[(bool, bool)]) {
        for &(a, b) in states {
            decoder.update(a, b).expect("valid transition");
        }
    }

    #[test]
    fn seeded_decoder_counts_four_per_cycle_and_knows_direction() {
        // Forward cycle: 00 -> 10 -> 11 -> 01 -> 00. Start from (true, true).
        let mut forward = Decoder::default();
        seed_decoder(&mut forward, true, true);
        walk(
            &mut forward,
            &[(false, true), (false, false), (true, false), (true, true)],
        );
        let mut backward = Decoder::default();
        seed_decoder(&mut backward, true, true);
        walk(
            &mut backward,
            &[(true, false), (false, false), (false, true), (true, true)],
        );

        assert_eq!(forward.counter().abs(), 4);
        assert_eq!(backward.counter(), -forward.counter());
    }

    #[test]
    fn rate_is_zero_once_the_window_has_passed_since_the_last_change() {
        let base = SystemTime::UNIX_EPOCH + Duration::from_secs(1_000);
        let mut state = MovementState::default();
        state.push(datapoint(base, 0, 0));
        state.push(datapoint(base, 100, 100));

        let now = base + Duration::from_millis(400);
        let rate = state
            .rate_counts_per_sec(now, Duration::from_millis(200))
            .unwrap();
        assert_eq!(rate, (0.0, 0.0));
    }

    #[test]
    fn rate_reflects_recent_movement() {
        let base = SystemTime::UNIX_EPOCH + Duration::from_secs(1_000);
        let mut state = MovementState::default();
        state.push(datapoint(base, 0, 0));
        state.push(datapoint(base, 150, 50));

        // 50 counts in the 200 ms window.
        let now = base + Duration::from_millis(200);
        let (a, b) = state
            .rate_counts_per_sec(now, Duration::from_millis(200))
            .unwrap();
        assert!((a - 250.0).abs() < 1e-9 && b == 0.0, "{a} {b}");
    }

    #[test]
    fn rate_just_after_startup_uses_the_history_available() {
        let base = SystemTime::UNIX_EPOCH + Duration::from_secs(1_000);
        let mut state = MovementState::default();
        state.push(datapoint(base, 0, 0));
        state.push(datapoint(base, 30, 10));

        // Only 50 ms of history, shorter than the 200 ms window.
        let now = base + Duration::from_millis(50);
        let (a, _) = state
            .rate_counts_per_sec(now, Duration::from_millis(200))
            .unwrap();
        assert!((a - 200.0).abs() < 1e-9, "{a}");
        assert!(MovementState::default()
            .rate_counts_per_sec(now, Duration::from_millis(200))
            .is_none());
    }

    #[test]
    fn circular_diff_wraps_around() {
        assert_eq!(circular_diff(COUNTS_PER_REV - 1, 1), 2);
        assert_eq!(circular_diff(10, 10), 0);
    }
}