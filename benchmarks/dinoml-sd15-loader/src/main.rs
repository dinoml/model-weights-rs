//! Fair host-loading comparison for `DinoML` SD1.5 constants.

use std::error::Error as StdError;

mod cli;
mod contract;
mod kyxc;
mod legacy_lane;
mod model_lane;
mod report;
mod runner;

fn main() -> Result<(), Box<dyn StdError>> {
    let arguments = cli::Arguments::parse()?;
    let report = runner::execute(&arguments)?;
    serde_json::to_writer_pretty(std::io::stdout().lock(), &report)?;
    println!();
    Ok(())
}
