// Copyright (c) 2024 Steven Rosenthal smr@dt3.org
// See LICENSE file in root directory for license terms.

use cedar_server::{cedar_server::server_main, encoder_tracker::EncoderTracker};
use pico_args::Arguments;
use std::sync::Arc;
use tokio::sync::Mutex;

fn main() {
    server_main(
        "Copyright (c) 2026 Steven Rosenthal smr@dt3.org.\n\
         Licensed for non-commercial use.\n\
         See LICENSE.md at https://github.com/smroid/cedar-server",
        // flutter_app_path=
        "../cedar/cedar-aim/cedar_flutter/build/web",
        // get_dependencies=
        |_pargs: Arguments| (None, None, Some(Arc::new(Mutex::new(EncoderTracker::new()))), None, None, None),
        // default_total_binning=
        None,
        // product_name_override=
        None,
    );
}
