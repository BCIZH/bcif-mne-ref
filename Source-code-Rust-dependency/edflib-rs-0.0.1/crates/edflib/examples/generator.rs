use anyhow::Result;
use edflib::*;

pub fn main() -> Result<()> {
    let path = "generator.edf";
    let edf = Edf::new(path.into());
    edf.open_file_writeonly().unwrap();
    Ok(())
}
