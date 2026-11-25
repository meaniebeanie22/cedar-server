//! Encoder-based dead reckoning for Cedar.
//!
//! An `ImuTrait` implementation that reads two quadrature encoders on a
//! background thread and keeps a short history of where they were, so that
//! plate solves (which arrive late) can be matched to the right encoder counts.
//!
//! This version sleeps until a GPIO edge interrupt arrives instead of polling.

use async_trait::async_trait;
use canonical_error::{CanonicalError, CanonicalErrorCode};
use cedar_elements::imu_trait::{
    HorizonCoordinates, ImuState, ImuTrait, TrackerState, TransformCalibration,
    ZeroBias,
};
use nalgebra::SMatrix;
use quadrature_decoder::{IncrementalDecoder, QuadStep};
use rppal::gpio::{Gpio, InputPin, Trigger};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::thread;
use std::time::{Duration, SystemTime};

// ---------------------------------------------------------------------------
// STEP 1: constants. Everything you might want to tune lives up here.
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

/// No count change for this long means the mount has stopped.
const MOVING_TIMEOUT: Duration = Duration::from_millis(20);

const REGRESSION_DATA_LEN: usize = 50;
const MIN_SOLVES_FOR_REGRESSION: usize = 3;
/// A new solve is ignored unless an encoder moved at least this many counts
/// since the last stored solve (0.5% of a revolution).
const MIN_COUNT_CHANGE: i64 = COUNTS_PER_REV / 200;

// ---------------------------------------------------------------------------
// STEP 2: data types.
// ---------------------------------------------------------------------------

type Decoder = IncrementalDecoder<QuadStep, i32>;

/// Encoder counts at one instant. Only recorded when the counts CHANGE.
#[derive(Clone, Copy)]
struct MovementDatapoint {
    time: SystemTime,
    a: i64,
    b: i64,
}

/// Everything the polling thread writes, grouped so ONE mutex protects it.
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
    platesolve_time: SystemTime,
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
// STEP 3: the tracker struct.
// ---------------------------------------------------------------------------

pub struct EncoderTracker {
    /// Parked here by `new()`, then moved out into the interrupt thread by
    /// `start()`. After that this is `None`. Order: A clk, A dt, B clk, B dt.
    pins: Mutex<Option<(Gpio, [InputPin; 4])>>,
    /// Written by the polling thread, read by the trait methods.
    movement: Arc<Mutex<MovementState>>,
    /// Front is new, back is old.
    platesolve_data: Mutex<VecDeque<PlatesolveDatapoint>>,
    regression: Mutex<Option<SMatrix<f64, 2, 2>>>,
    /// Tells the polling thread to keep going (true) or exit (false).
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
            regression: Mutex::new(None),
            running: Arc::new(AtomicBool::new(false)),
        }
    }

    fn do_platesolve_regression(
        &self,
    ) -> Result<SMatrix<f64, 2, 2>, CanonicalError> {
        let solves = self.platesolve_data.lock().map_err(poisoned)?;
        if solves.len() < MIN_SOLVES_FOR_REGRESSION {
            return Err(failed_precondition("Not enough platesolves yet"));
        }

        // TODO: fit `fitted` from `solves` here, then store and return it:
        //   *self.regression.lock().map_err(poisoned)? = Some(fitted);
        //   Ok(fitted)
        Err(failed_precondition("Platesolve regression not implemented yet"))
    }
}

/// When the tracker is dropped, tell the polling thread to exit.
impl Drop for EncoderTracker {
    fn drop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
    }
}

// ---------------------------------------------------------------------------
// STEP 5: the interrupt loop. Runs on its own thread and owns the four pins.
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

        let now =
            (i64::from(decoder_a.counter()), i64::from(decoder_b.counter()));
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
    // STEP 4: spawns the interrupt thread. Moves the pins into it.
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

    // STEP 6: look up the encoder counts at the time of the solve.
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
                platesolve_time: *timestamp,
                encoder_a_count: a,
                encoder_b_count: b,
            },
        );
    }

    // No plate solution is available. This is either due to visual obstruction
    // (clouds, etc) or platform motion preventing star detection.
    async fn report_camera_pointing_lost(&self) {}

    // Discards all calibration state. The raw movement history is a
    // measurement, not calibration, so it stays.
    async fn reset(&self) {
        *self
            .regression
            .lock()
            .expect("Couldn't reset regression matrix") = None;
        self.platesolve_data
            .lock()
            .expect("Couldn't reset platesolve_data")
            .clear();
    }

    async fn get_estimated_camera_pointing(
        &self,
        _timestamp: &SystemTime,
    ) -> Result<HorizonCoordinates, CanonicalError> {
        // Copy the cached value out so the lock is released before the
        // regression (which locks again) can run.
        let cached = *self.regression.lock().map_err(poisoned)?;
        let _regression = match cached {
            Some(regression) => regression,
            None => self.do_platesolve_regression()?,
        };

        // TODO: turn the current counts + regression into coordinates.
        Ok(HorizonCoordinates {
            zenith_roll_angle: 0f64,
            altitude: 0f64,
            azimuth: 0f64,
        })
    }

    // STEP 7: tracker state comes from when the counts last changed.
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

    // There is no raw IMU here, so report that instead of panicking.
    async fn get_state(
        &self,
    ) -> Result<(ImuState, SystemTime), CanonicalError> {
        Err(failed_precondition("EncoderTracker has no raw IMU readings"))
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
        Ok((0f64, SystemTime::now()))
    }

    fn get_model(&self) -> String {
        String::from("EncoderTracker")
    }

    fn save_state(&self) -> Result<(), CanonicalError> {
        Ok(())
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
    fn circular_diff_wraps_around() {
        assert_eq!(circular_diff(COUNTS_PER_REV - 1, 1), 2);
        assert_eq!(circular_diff(10, 10), 0);
    }
}
