use async_trait::async_trait;
use canonical_error::CanonicalError;
use cedar_elements::imu_trait::{
    AccelData, GyroData, HorizonCoordinates, ImuState, ImuTrait, TrackerState,
    TransformCalibration, ZeroBias,
};
use nalgebra::{DMatrix, SMatrix};
use quadrature_encoder::{Async, Rotary};
use quadrature_encoder::{IncrementalEncoder, QuadStep};
use rppal::gpio::{Gpio, InputPin};
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;
use std::time::SystemTime;

const REGRESSION_DATA_LEN: usize = 50;
const MOVEMENT_DATA_LEN: usize = 20;
const ENCODER_PINS: (u8, u8, u8, u8) = (1, 2, 3, 4); // Encoder A: A, B; Encoder B: A, B
const MOVING_TIMEOUT: usize = 20; // millis the
const PLATESOLVE_COUNT_TOLERANCE: f64 = 0.005;
const COUNTS_PER_REV: usize = 4000;

type Encoder = IncrementalEncoder<Rotary, InputPin, InputPin, QuadStep>; 
pub struct EncoderTracker {
    encoder_a: Encoder,
    encoder_b: Encoder,
    platesolve_data: Arc<Mutex<VecDeque<PlatesolveDatapoint>>>, // front is new, back is old
    movement_data: Arc<Mutex<VecDeque<MovementDatapoint>>>, // front is new, back is old
    regression: Arc<Mutex<Option<SMatrix<f64, 2, 2>>>>,
}

impl EncoderTracker {
    pub fn new() -> EncoderTracker {
        let gpio = Gpio::new().expect("Couldn't init a gpio");

        EncoderTracker {
            encoder_a: Encoder::new(
                gpio.get(ENCODER_PINS.0)
                    .expect("Couldn't init an encoder pin")
                    .into_input(),
                gpio.get(ENCODER_PINS.1)
                    .expect("Couldn't init an encoder pin")
                    .into_input(),
            ),
            encoder_b: Encoder::new(
                gpio.get(ENCODER_PINS.2)
                    .expect("Couldn't init an encoder pin")
                    .into_input(),
                gpio.get(ENCODER_PINS.3)
                    .expect("Couldn't init an encoder pin")
                    .into_input(),
            ),
            platesolve_data: Arc::new(Mutex::new(VecDeque::with_capacity(
                REGRESSION_DATA_LEN,
            ))),
            movement_data: Arc::new(Mutex::new(VecDeque::with_capacity(
                MOVEMENT_DATA_LEN,
            ))),
            regression: Arc::new(Mutex::new(None)),
        }
    }

    fn do_platesolve_regression(&self) -> Result<SMatrix<f64, 2, 2>, CanonicalError> {
        // do regression

        // save and return
        let mut regression_store = self.regression.lock();
        self.regression.lock()? = regression;
        Ok(regression)
    }
    /// Returns raw encoder counts, whilst also appending them to the buffer
    fn get_counts(&self) -> Result<(i64, i64), ()> {
        unimplemented!();

        let encoder_a_count = 0;
        let encoder_b_count = 0;

        EncoderTracker::add_to_buffer(
            &mut self
                .movement_data
                .lock()
                .expect("Couldn't get access to the movement buffer"),
            MOVEMENT_DATA_LEN,
            MovementDatapoint {
                movement_time: SystemTime::now(),
                encoder_a_count,
                encoder_b_count,
            },
        );
        Ok((encoder_a_count, encoder_b_count))
    }

    fn add_to_buffer<T>(buffer: &mut VecDeque<T>, max_size: usize, element: T) {
        buffer.push_front(element);

        if buffer.len() > max_size {
            buffer.pop_back();
        }
    }
}

#[derive(Clone, Copy)]
struct PlatesolveDatapoint {
    platesolve_result: HorizonCoordinates,
    platesolve_time: SystemTime,
    encoder_a_count: u64, // normalised
    encoder_b_count: u64,
}

#[derive(Clone, Copy)]
struct MovementDatapoint {
    movement_time: SystemTime,
    encoder_a_count: i64,
    encoder_b_count: i64,
}

#[async_trait]
impl ImuTrait for EncoderTracker {
    // Spawns the background worker. Must be called from within a Tokio runtime
    // context, after new().
    fn start(&self) {}

    // Conveys information obtained from plate solving to the IMU fusion
    // algorithms. The timestamp must be strictly non decreasing for successive
    // calls.
    async fn report_true_camera_pointing(
        &self,
        camera_pointing: &HorizonCoordinates,
        timestamp: &SystemTime,
    ) {
        let mut platesolve_buffer = self
            .platesolve_data
            .lock()
            .expect("Unable to lock platesolve_data to add new solve");

        // find the counts immediately before the timestamp of the solve
        let movement_buffer = self.movement_data.lock().expect("Unable to lock movement_data to lookup encoder readings when adding a solve");

        let movement = movement_buffer
            .iter()
            .filter(|movement_datapoint: &&MovementDatapoint| {
                match timestamp.duration_since(movement_datapoint.movement_time)
                {
                    Ok(duration) => true,
                    Err(_) => false,
                }
            })
            .next();

        let movement = match movement {
            Some(movement) => {
                (movement.encoder_a_count, movement.encoder_b_count)
            }
            None => {
                log::warn!(
                    "No prior counts to the platesolve, went and got one."
                );
                match self.get_counts() {
                    Ok(x) => x,
                    Err(_) => return,
                }
            }
        }; // gets the most recent set of counts that happened prior to the solve

        // compare counts against most recent solve datapoint
        if let Some(most_recent_solve) = platesolve_buffer.front() {
            if ((most_recent_solve.encoder_a_count.abs_diff(
                movement.0.rem_euclid(const { COUNTS_PER_REV as i64 }) as u64,
            )) < const {
                (PLATESOLVE_COUNT_TOLERANCE * (COUNTS_PER_REV as f64)) as u64
            } && (most_recent_solve.encoder_b_count.abs_diff(
                movement.1.rem_euclid(const { COUNTS_PER_REV as i64 }) as u64,
            )) < const {
                (PLATESOLVE_COUNT_TOLERANCE * (COUNTS_PER_REV as f64)) as u64
            }) {
                return; // not different enough to care
            }
        }; // if not big enough count change ignore

        // add new measurement
        EncoderTracker::add_to_buffer(
            &mut platesolve_buffer,
            REGRESSION_DATA_LEN,
            PlatesolveDatapoint {
                platesolve_result: *camera_pointing,
                platesolve_time: *timestamp,
                encoder_a_count: movement.0.rem_euclid(COUNTS_PER_REV as i64)
                    as u64,
                encoder_b_count: movement.1.rem_euclid(COUNTS_PER_REV as i64)
                    as u64,
            },
        );
    }

    // No plate solution is available. This is either due to visual obstruction
    // (clouds, etc) or platform motion preventing star detection.
    async fn report_camera_pointing_lost(&self) {}

    // Discards all calibration state.
    async fn reset(&self) {
        *self
            .regression
            .lock()
            .expect("Couldn't reset regression matrix") = None;
        *self
            .movement_data
            .lock()
            .expect("Couldn't reset movement_data") =
            VecDeque::with_capacity(MOVEMENT_DATA_LEN);
        *self
            .platesolve_data
            .lock()
            .expect("Couldn't reset platesolve_data") =
            VecDeque::with_capacity(REGRESSION_DATA_LEN);
    }

    // IMU-derived estimate of camera pointing as of the given time. The
    // timestamp must not precede the timestamp of the most recent
    // report_true_camera_pointing() call.
    async fn get_estimated_camera_pointing(
        &self,
        timestamp: &SystemTime,
    ) -> Result<HorizonCoordinates, CanonicalError> {
        let regression = self.regression.lock();

        let regression = match regression {
            Ok(x) => x,
            Err(y) => return Err(CanonicalError { code: (canonical_error::CanonicalErrorCode::FailedPrecondition), message: (y.to_string()) }),
        };

        let regression = match *regression {
            Some(x) => x,
            None => self.do_platesolve_regression()?
        };

        let roll_angle = 0f64;
        let alt = 0f64;
        let az = 0f64;

        let coords = HorizonCoordinates {
            zenith_roll_angle: roll_angle,
            altitude: alt,
            azimuth: az
        };
        Ok(coords)
    }

    // Returns the current state of the IMU tracker logic.
    async fn get_tracker_state(&self) -> TrackerState {
        let Ok(movement_data) = self.movement_data.lock() else {
            return TrackerState::Lost;
        };

        let Some(movement_datapoint) = movement_data.front() else {
            return TrackerState::Lost;
        };

        let Ok(time_since_last_move) =
            SystemTime::now().duration_since(movement_datapoint.movement_time)
        else {
            return TrackerState::Lost;
        };

        if time_since_last_move.as_nanos()
            < const { (MOVING_TIMEOUT as u128) * 1000 * 1000 }
        {
            return TrackerState::Moving;
        } else {
            return TrackerState::Motionless;
        };
    }

    // Returns the current calibration state, if any, of the IMU tracker.
    async fn get_calibration(
        &self,
    ) -> (Option<ZeroBias>, Option<TransformCalibration>) {
        (None, None)
    }

    // Returns the most recent raw IMU reading.
    async fn get_state(
        &self,
    ) -> Result<(ImuState, SystemTime), CanonicalError> {
        unimplemented!();
        let x = 0f64;
        let y = 0f64;
        let time = SystemTime::now();

        Ok((
            ImuState {
                timestamp: SystemTime::now(),
                accel: AccelData {
                    x: 0f64,
                    y: 0f64,
                    z: 0f64,
                },
                gyro: GyroData { x, y, z: 0f64 }, //x is az, y is alt
            },
            time,
        ))
    }

    // Returns the bias-adjusted angular velocity magnitude (degrees/s) seen in
    // the most recent samples. Returns failed_precondition error if there is no
    // IMU calibration.
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

    // Returns the IMU's model.
    fn get_model(&self) -> String {
        String::from("EncoderTracker")
    }

    // Saves the IMU's calibrated state. This call blocks on IO.
    fn save_state(&self) -> Result<(), CanonicalError> {
        Ok(())
    }
}
