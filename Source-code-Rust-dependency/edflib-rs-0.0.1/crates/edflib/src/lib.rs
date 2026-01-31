use std::{ ops::Deref, os::raw::c_int, path::PathBuf, sync::{ Arc, Mutex } };

use anyhow::{ anyhow, Result };
use derive_new::new;
use edflib_sys::*;
use utils::*;

mod utils;

pub enum Filetype {
    EDF,
    BDF,
}

impl Filetype {
    fn from(ext: &str) -> Self {
        match ext {
            "edf" => Filetype::EDF,
            "bdf" => Filetype::BDF,
            _ => Filetype::EDF,
        }
    }
    fn as_str(&self) -> &str {
        match self {
            Filetype::EDF => "edf",
            Filetype::BDF => "bdf",
        }
    }
}

pub enum AnnotationPosition {
    Start,
    Middle,
    End,
}

impl AnnotationPosition {
    fn to_raw(&self) -> i32 {
        (match self {
            AnnotationPosition::Start => EDF_ANNOT_IDX_POS_START,
            AnnotationPosition::Middle => EDF_ANNOT_IDX_POS_MIDDLE,
            AnnotationPosition::End => EDF_ANNOT_IDX_POS_END,
        }) as i32
    }
}

#[derive(new)]
struct Inner {
    #[new(value = "0")]
    hdl: i32,
    #[new(value = "Filetype::EDF")]
    filetype: Filetype,
}

#[derive(new)]
pub struct Edf {
    path: PathBuf,
    #[new(value = "Arc::new(Mutex::new(Inner::new()))")]
    inner: Arc<Mutex<Inner>>,
    #[new(value = "1")]
    pub number_of_signals: i32,
}

impl Edf {
    fn get_hdl(&self) -> i32 {
        let inner = self.inner.lock().unwrap();
        inner.hdl
    }
    pub fn open_file_writeonly(&self) -> Result<()> {
        let path = PathBuf::from(self.path.to_str().unwrap());
        let ext = path.extension().unwrap().to_str().unwrap();
        let filetype = Filetype::from(ext);

        let path = str_to_char(path.to_str().unwrap());
        let mut inner = self.inner.lock().unwrap();

        let filetype = match filetype {
            Filetype::EDF => EDFLIB_FILETYPE_EDFPLUS as c_int,
            Filetype::BDF => EDFLIB_FILETYPE_BDFPLUS as c_int,
        };
        let hdl = unsafe { edfopen_file_writeonly(path, filetype, self.number_of_signals) };
        inner.hdl = hdl;

        if hdl < 0 {
            let msg = format!("Can not open file \"{}\"for writing", self.path.to_str().unwrap());
            Err(anyhow!(msg))
        } else {
            Ok(())
        }
    }

    pub fn finish(&self) -> Result<()> {
        let result = unsafe { edfclose_file(self.get_hdl()) };

        if result < 0 {
            Err(anyhow!("Error finishing and closing the file"))
        } else {
            Ok(())
        }
    }

    pub fn set_samplefrequency(&self, edfsignal: i32, samplefrequency: i32) -> Result<()> {
        let result = unsafe { edf_set_samplefrequency(self.get_hdl(), edfsignal, samplefrequency) };

        if result < 0 {
            Err(anyhow!("Error setting set_samplefrequency"))
        } else {
            Ok(())
        }
    }

    pub fn set_digital_maximum(&self, edfsignal: i32, dig_max: i32) -> Result<()> {
        let result = unsafe { edf_set_digital_maximum(self.get_hdl(), edfsignal, dig_max) };

        if result < 0 {
            Err(anyhow!("Error setting set_digital_maximum"))
        } else {
            Ok(())
        }
    }

    pub fn set_digital_minimum(&self, edfsignal: i32, dig_min: i32) -> Result<()> {
        let result = unsafe { edf_set_digital_minimum(self.get_hdl(), edfsignal, dig_min) };

        if result < 0 {
            Err(anyhow!("Error setting set_digital_minimum"))
        } else {
            Ok(())
        }
    }

    pub fn set_physical_dimension(&self, edfsignal: i32, phys_dim: String) -> Result<()> {
        let phys_dim = str_to_char(phys_dim.as_str());
        let result = unsafe { edf_set_physical_dimension(self.get_hdl(), edfsignal, phys_dim) };

        if result < 0 {
            Err(anyhow!("Error setting set_physical_dimension"))
        } else {
            Ok(())
        }
    }

    pub fn set_label(&self, edfsignal: i32, label: String) -> Result<()> {
        let label = str_to_char(label.as_str());
        let result = unsafe { edf_set_label(self.get_hdl(), edfsignal, label) };

        if result < 0 {
            Err(anyhow!("Error setting set_label"))
        } else {
            Ok(())
        }
    }

    pub fn set_equipment(&self, equipment: String) -> Result<()> {
        let equipment = str_to_char(equipment.as_str());
        let result = unsafe { edf_set_equipment(self.get_hdl(), equipment) };

        if result < 0 {
            Err(anyhow!("Error setting set_equipment"))
        } else {
            Ok(())
        }
    }

    pub fn set_annot_chan_idx_pos(&self, position: AnnotationPosition) -> Result<()> {
        let result = unsafe { edf_set_annot_chan_idx_pos(self.get_hdl(), position.to_raw()) };

        if result < 0 {
            Err(anyhow!("Error setting set_annot_chan_idx_pos"))
        } else {
            Ok(())
        }
    }

    pub fn set_number_of_annotation_signals(&self, annot_signals: usize) -> Result<()> {
        let result = unsafe {
            edf_set_number_of_annotation_signals(self.get_hdl(), annot_signals as i32)
        };

        if result < 0 {
            Err(anyhow!("Error setting set_number_of_annotation_signals"))
        } else {
            Ok(())
        }
    }

    pub fn write_samples(&self, samples: &mut Vec<f64>) -> Result<()> {
        let buf: *mut f64 = samples.as_mut_ptr().cast::<f64>();
        let result = unsafe { edfwrite_physical_samples(self.get_hdl(), buf) };

        if result < 0 {
            Err(anyhow!("Error write_samples"))
        } else {
            Ok(())
        }
    }

    pub fn write_annotation(&self, onset: i64, duration: i64, description: String) -> Result<()> {
        let description = str_to_char(description.as_str());
        let result = unsafe {
            edfwrite_annotation_latin1_hr(self.get_hdl(), onset, duration, description)
        };

        if result < 0 {
            Err(anyhow!("Error write_annotation"))
        } else {
            Ok(())
        }
    }
}
