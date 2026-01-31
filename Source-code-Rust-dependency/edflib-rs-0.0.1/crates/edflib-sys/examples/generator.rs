/*
reimplementation of https://gitlab.com/Teuniz/EDFlib/-/blob/master/test_generator.c in Rust
this program generates an EDFplus or BDFplus testfile with the following signals:

signal label/waveform     amplitude    f       sf
------------------------------------------------------
    1    squarewave           100 uV    0.1Hz   200 Hz
    2    ramp                 100 uV    1 Hz    200 Hz
    3    pulse 1              100 uV    1 Hz    200 Hz
    4    pulse 2              100 uV    1 Hz    256 Hz
    5    pulse 3              100 uV    1 Hz    217 Hz
    6    noise                100 uV    - Hz    200 Hz
    7    sine 1 Hz            100 uV    1 Hz    200 Hz
    8    sine 8 Hz + DC       100 uV    8 Hz    200 Hz
    9    sine 8.1777 Hz + DC  100 uV    8.25 Hz 200 Hz
   10    sine 8.5 Hz          100 uV    8.5Hz   200 Hz
   11    sine 15 Hz           100 uV   15 Hz    200 Hz
   12    sine 17 Hz           100 uV   17 Hz    200 Hz
   13    sine 50 Hz           100 uV   50 Hz    200 Hz
   14    DC event 8-bits code 1 V   100 mS/bit  200 Hz

*/

use std::{ f64::consts::PI, ffi::{ c_int, c_long, CString }, os::raw::c_char };

use edflib_sys::*;

const SMP_FREQ: i32 = 200;
const SMP_FREQ_2: i32 = 256;
const SMP_FREQ_3: i32 = 217;
const FILE_DURATION: i64 = 600;
const BDF_FORMAT: bool = false;
const TRIGGERS_SIZE: usize = 512;
const BUFFER_SIZE: usize = 1_000;
const OUTPUT_NAME: &str = "generator";

// https://en.cppreference.com/w/cpp/numeric/random/RAND_MAX
pub const RAND_MAX: c_int = 2_147_483_647;

#[link(name = "c")]
extern "C" {
    fn rand() -> i32;
    // fn srand(seed: u32);
}

fn str_to_char(input: &str) -> *const c_char {
    CString::new(input).unwrap().into_raw()
}

struct EventStats {
    samples: c_long,
    triggers: [c_long; TRIGGERS_SIZE],
    index: c_int,
    code: c_int,
    bitposition: c_int,
    smp_in_bit: c_int,
}

pub fn main() {
    let triggers: [c_long; TRIGGERS_SIZE] = (0..TRIGGERS_SIZE)
        .map(|index| (index * 1667 + 1951) as c_long)
        .collect::<Vec<_>>()
        .try_into()
        .expect("wrong size iterator");

    let mut dc_event_stat = EventStats {
        samples: 0,
        triggers,
        index: 0,
        code: 0,
        bitposition: 0,
        smp_in_bit: 0,
    };

    let number_of_signals = 14;
    let mut buf: [f64; BUFFER_SIZE] = (0..BUFFER_SIZE)
        .map(|_| 0.0)
        .collect::<Vec<_>>()
        .try_into()
        .expect("wrong size iterator");
    let (ext, filetype) = match BDF_FORMAT {
        true => ("bdf", EDFLIB_FILETYPE_BDFPLUS as c_int),
        false => ("edf", EDFLIB_FILETYPE_EDFPLUS as c_int),
    };
    let path = format!("./{}.{}", OUTPUT_NAME, ext);
    println!("Creating {path}");
    let path = str_to_char(path.as_str());

    let hdl = unsafe { edfopen_file_writeonly(path, filetype, number_of_signals) };

    if hdl < 0 {
        return println!("error: edfopen_file_writeonly() {hdl}");
    }

    for i in 0..number_of_signals {
        if (unsafe { edf_set_samplefrequency(hdl, i, SMP_FREQ) }) > 0 {
            return println!("error: edf_set_samplefrequency()");
        }
    }

    if (unsafe { edf_set_samplefrequency(hdl, 3, SMP_FREQ_2) }) > 0 {
        return println!("error: edf_set_samplefrequency()");
    }

    if (unsafe { edf_set_samplefrequency(hdl, 4, SMP_FREQ_3) }) > 0 {
        return println!("error: edf_set_samplefrequency()");
    }

    if (unsafe { edf_set_samplefrequency(hdl, 13, 1_000) }) > 0 {
        return println!("error: edf_set_samplefrequency()");
    }

    if BDF_FORMAT {
        for i in 0..number_of_signals {
            if (unsafe { edf_set_digital_maximum(hdl, i, 8_388_607) }) > 0 {
                return println!("error: edf_set_digital_maximum()");
            }
        }

        for i in 0..number_of_signals {
            if (unsafe { edf_set_digital_minimum(hdl, i, -8_388_608) }) > 0 {
                return println!("error: edf_set_digital_minimum()");
            }
        }

        if (unsafe { edf_set_digital_minimum(hdl, 13, 8_300_000) }) > 0 {
            return println!("error: edf_set_digital_minimum()");
        }
    } else {
        for i in 0..number_of_signals {
            if (unsafe { edf_set_digital_maximum(hdl, i, 32_767) }) > 0 {
                return println!("error: edf_set_digital_maximum()");
            }
        }

        for i in 0..number_of_signals {
            if (unsafe { edf_set_digital_minimum(hdl, i, -32_768) }) > 0 {
                return println!("error: edf_set_digital_minimum()");
            }
        }
    }

    for i in 0..number_of_signals {
        if (unsafe { edf_set_physical_maximum(hdl, i, 1_000.0) }) > 0 {
            return println!("error: edf_set_physical_maximum()");
        }
    }

    if (unsafe { edf_set_physical_maximum(hdl, 8, 262_143.0) }) > 0 {
        return println!("error: edf_set_physical_maximum()");
    }

    if (unsafe { edf_set_physical_maximum(hdl, 13, 10.0) }) > 0 {
        return println!("error: edf_set_physical_maximum()");
    }

    for i in 0..number_of_signals {
        if (unsafe { edf_set_physical_minimum(hdl, i, -1_000.0) }) > 0 {
            return println!("error: edf_set_physical_minimum()");
        }
    }

    if (unsafe { edf_set_physical_minimum(hdl, 8, -262_144.0) }) > 0 {
        return println!("error: edf_set_physical_minimum()");
    }

    if (unsafe { edf_set_physical_minimum(hdl, 13, -10.0) }) > 0 {
        return println!("error: edf_set_physical_minimum()");
    }

    for i in 0..number_of_signals {
        if (unsafe { edf_set_physical_dimension(hdl, i, str_to_char("uV")) }) > 0 {
            return println!("error: edf_set_physical_dimension()");
        }
    }

    if (unsafe { edf_set_physical_dimension(hdl, 13, str_to_char("V")) }) > 0 {
        return println!("error: edf_set_physical_dimension()");
    }

    let mut i: c_int = 0;

    i += 1;
    if (unsafe { edf_set_label(hdl, i, str_to_char("squarewave")) }) > 0 {
        return println!("error: edf_set_label()");
    }

    i += 1;
    if (unsafe { edf_set_label(hdl, i, str_to_char("ramp")) }) > 0 {
        return println!("error: edf_set_label()");
    }

    i += 1;
    if (unsafe { edf_set_label(hdl, i, str_to_char("pulse 1")) }) > 0 {
        return println!("error: edf_set_label()");
    }

    i += 1;
    if (unsafe { edf_set_label(hdl, i, str_to_char("pulse 2")) }) > 0 {
        return println!("error: edf_set_label()");
    }

    i += 1;
    if (unsafe { edf_set_label(hdl, i, str_to_char("pulse 3")) }) > 0 {
        return println!("error: edf_set_label()");
    }

    i += 1;
    if (unsafe { edf_set_label(hdl, i, str_to_char("noise")) }) > 0 {
        return println!("error: edf_set_label()");
    }

    i += 1;
    if (unsafe { edf_set_label(hdl, i, str_to_char("sine 1 Hz")) }) > 0 {
        return println!("error: edf_set_label()");
    }

    i += 1;
    if (unsafe { edf_set_label(hdl, i, str_to_char("sine 8 Hz + DC")) }) > 0 {
        return println!("error: edf_set_label()");
    }

    i += 1;
    if (unsafe { edf_set_label(hdl, i, str_to_char("sine 8.1777 Hz + DC")) }) > 0 {
        return println!("error: edf_set_label()");
    }

    i += 1;
    if (unsafe { edf_set_label(hdl, i, str_to_char("sine 8.5 Hz")) }) > 0 {
        return println!("error: edf_set_label()");
    }

    i += 1;
    if (unsafe { edf_set_label(hdl, i, str_to_char("sine 15 Hz")) }) > 0 {
        return println!("error: edf_set_label()");
    }

    i += 1;
    if (unsafe { edf_set_label(hdl, i, str_to_char("sine 17 Hz")) }) > 0 {
        return println!("error: edf_set_label()");
    }

    i += 1;
    if (unsafe { edf_set_label(hdl, i, str_to_char("sine 50 Hz")) }) > 0 {
        return println!("error: edf_set_label()");
    }

    i += 1;
    if (unsafe { edf_set_label(hdl, i, str_to_char("DC 01")) }) > 0 {
        return println!("error: edf_set_label()");
    }

    if (unsafe { edf_set_equipment(hdl, str_to_char("test generator")) }) > 0 {
        return println!("error: edf_set_equipment()");
    }

    unsafe {
        edf_set_birthdate(hdl, 1969, 6, 30);
    }

    if (unsafe { edf_set_annot_chan_idx_pos(hdl, EDF_ANNOT_IDX_POS_MIDDLE as i32) }) > 0 {
        return println!("error: edf_set_annot_chan_idx_pos()");
    }

    if (unsafe { edf_set_number_of_annotation_signals(hdl, 2) }) > 0 {
        return println!("error: edf_set_number_of_annotation_signals()");
    }

    let mut sine_1 = 0.0;
    let mut sine_8 = 0.0;
    let mut sine_81777 = 0.0;
    let mut sine_85 = 0.0;
    let mut sine_15 = 0.0;
    let mut sine_17 = 0.0;
    let mut sine_50 = 0.0;

    for j in 0..FILE_DURATION {
        if j % 10 < 5 /* square */ {
            for i in 0..SMP_FREQ {
                buf[i as usize] = 100.0;
            }
        } else {
            for i in 0..SMP_FREQ {
                buf[i as usize] = -100.0;
            }
        }

        if (unsafe { edfwrite_physical_samples(hdl, buf.as_mut_ptr()) }) > 0 {
            return println!("error: edfwrite_physical_samples()");
        }

        for i in 0..SMP_FREQ /* ramp */ {
            buf[i as usize] = -100.0 + (i as f64) * (200.0 / (SMP_FREQ as f64));
        }

        if (unsafe { edfwrite_physical_samples(hdl, buf.as_mut_ptr()) }) > 0 {
            return println!("error: edfwrite_physical_samples()");
        }

        for i in 0..SMP_FREQ /* pulse 1 */ {
            buf[i as usize] = 0.0;
        }

        buf[0] = 100.0;

        buf[(SMP_FREQ - 2) as usize] = 100.0;

        if (unsafe { edfwrite_physical_samples(hdl, buf.as_mut_ptr()) }) > 0 {
            return println!("error: edfwrite_physical_samples()");
        }

        for i in 0..SMP_FREQ_2 /* pulse 2 */ {
            buf[i as usize] = 0.0;
        }

        buf[0] = 100.0;

        buf[(SMP_FREQ_2 - 2) as usize] = 100.0;

        if (unsafe { edfwrite_physical_samples(hdl, buf.as_mut_ptr()) }) > 0 {
            return println!("error: edfwrite_physical_samples()");
        }

        for i in 0..SMP_FREQ_3 /* pulse 3 */ {
            buf[i as usize] = 0.0;
        }

        buf[0] = 100.0;

        buf[(SMP_FREQ_3 - 2) as usize] = 100.0;

        if (unsafe { edfwrite_physical_samples(hdl, buf.as_mut_ptr()) }) > 0 {
            return println!("error: edfwrite_physical_samples()");
        }

        for i in 0..SMP_FREQ /* noise */ {
            buf[i as usize] = (100.0 *
                (((unsafe { rand() }) as f64) / ((RAND_MAX as f64) + 1.0))) as f64;
        }

        if (unsafe { edfwrite_physical_samples(hdl, buf.as_mut_ptr()) }) > 0 {
            return println!("error: edfwrite_physical_samples()");
        }

        for i in 0..SMP_FREQ /* sine 1 Hz */ {
            let mut q = PI * 2.0;
            q /= SMP_FREQ as f64;
            sine_1 += q;
            q = f64::sin(sine_1);
            q *= 100.0;
            buf[i as usize] = q;
        }

        if (unsafe { edfwrite_physical_samples(hdl, buf.as_mut_ptr()) }) > 0 {
            return println!("error: edfwrite_physical_samples()");
        }

        for i in 0..SMP_FREQ /* sine 8 Hz */ {
            let mut q = PI * 2.0;
            q /= (SMP_FREQ as f64) / 8.0;
            sine_8 += q;
            q = f64::sin(sine_8);
            q *= 100.0;
            buf[i as usize] = q + 800.0; /* add dc-offset */
        }

        if (unsafe { edfwrite_physical_samples(hdl, buf.as_mut_ptr()) }) > 0 {
            return println!("error: edfwrite_physical_samples()");
        }

        for i in 0..SMP_FREQ /* sine 8.1777 Hz */ {
            let mut q = PI * 2.0;
            q /= (SMP_FREQ as f64) / 8.1777;
            sine_81777 += q;
            q = f64::sin(sine_81777);
            q *= 100.0;
            buf[i as usize] = q + 6000.0; /* add dc-offset */
        }

        if (unsafe { edfwrite_physical_samples(hdl, buf.as_mut_ptr()) }) > 0 {
            return println!("error: edfwrite_physical_samples()");
        }

        for i in 0..SMP_FREQ /* sine 8.5 Hz */ {
            let mut q = PI * 2.0;
            q /= (SMP_FREQ as f64) / 8.5;
            sine_85 += q;
            q = f64::sin(sine_85);
            q *= 100.0;
            buf[i as usize] = q;
        }

        if (unsafe { edfwrite_physical_samples(hdl, buf.as_mut_ptr()) }) > 0 {
            return println!("error: edfwrite_physical_samples()");
        }

        for i in 0..SMP_FREQ /* sine 15 Hz */ {
            let mut q = PI * 2.0;
            q /= (SMP_FREQ as f64) / 15.0;
            sine_15 += q;
            q = f64::sin(sine_15);
            q *= 100.0;
            buf[i as usize] = q;
        }

        if (unsafe { edfwrite_physical_samples(hdl, buf.as_mut_ptr()) }) > 0 {
            return println!("error: edfwrite_physical_samples()");
        }

        for i in 0..SMP_FREQ /* sine 17 Hz */ {
            let mut q = PI * 2.0;
            q /= (SMP_FREQ as f64) / 17.0;
            sine_17 += q;
            q = f64::sin(sine_17);
            q *= 100.0;
            buf[i as usize] = q;
        }

        if (unsafe { edfwrite_physical_samples(hdl, buf.as_mut_ptr()) }) > 0 {
            return println!("error: edfwrite_physical_samples()");
        }

        for i in 0..SMP_FREQ /* sine 50 Hz */ {
            let mut q = PI * 2.0;
            q /= (SMP_FREQ as f64) / 50.0;
            sine_50 += q;
            q = f64::sin(sine_50);
            q *= 100.0;
            buf[i as usize] = q;
        }

        if (unsafe { edfwrite_physical_samples(hdl, buf.as_mut_ptr()) }) > 0 {
            return println!("error: edfwrite_physical_samples()");
        }

        for i in 0..1000 /* DC 01 */ {
            if dc_event_stat.bitposition > 0 {
                if dc_event_stat.bitposition == 1 {
                    buf[i as usize] = 1.0;
                } else {
                    if dc_event_stat.code & (1 << (dc_event_stat.bitposition - 2)) > 0 {
                        buf[i as usize] = 1.0;
                    } else {
                        buf[i as usize] = 0.0;
                    }
                }

                dc_event_stat.smp_in_bit += 1;
                if dc_event_stat.smp_in_bit >= 10 {
                    dc_event_stat.smp_in_bit = 0;
                    dc_event_stat.bitposition += 1;
                }

                if dc_event_stat.bitposition > 10 {
                    dc_event_stat.bitposition = 0;
                    dc_event_stat.smp_in_bit = 0;
                    dc_event_stat.code += 1;
                    dc_event_stat.code &= 255;

                    dc_event_stat.index += 1;
                    if dc_event_stat.index >= 512 {
                        dc_event_stat.index = 0;
                        dc_event_stat.code = 0;
                    }
                }
            } else {
                if dc_event_stat.samples == dc_event_stat.triggers[dc_event_stat.index as usize] {
                    /*          edfwrite_annotation_latin1(hdl, dc_event_stat.samples * 10LL, -1LL, "Trigger");  */

                    dc_event_stat.bitposition = 1;
                    dc_event_stat.smp_in_bit = 1;
                    buf[i as usize] = 1.0;
                } else {
                    buf[i as usize] = 0.0;
                }
            }

            dc_event_stat.samples += 1;
        }

        if (unsafe { edfwrite_physical_samples(hdl, buf.as_mut_ptr()) }) > 0 {
            return println!("error: edfwrite_physical_samples()");
        }
    }

    unsafe {
        edfwrite_annotation_latin1_hr(hdl, 0, -1, str_to_char("Recording starts"));
        edfwrite_annotation_latin1_hr(hdl, 298_000_000, -1, str_to_char("Test 1"));
        edfwrite_annotation_latin1_hr(
            hdl,
            294_000_000 + ((1_000_000.0 / (SMP_FREQ as f64)) as i64) * ((SMP_FREQ - 2) as i64),
            -1,
            str_to_char("pulse 1")
        );
        edfwrite_annotation_latin1_hr(
            hdl,
            295_000_000 + ((1_000_000.0 / (SMP_FREQ_2 as f64)) as i64) * ((SMP_FREQ_2 - 2) as i64),
            -1,
            str_to_char("pulse 2")
        );
        edfwrite_annotation_latin1_hr(
            hdl,
            296_000_000 + ((1_000_000.0 / (SMP_FREQ_3 as f64)) as i64) * ((SMP_FREQ_3 - 2) as i64),
            -1,
            str_to_char("pulse 3")
        );
        edfwrite_annotation_latin1_hr(
            hdl,
            FILE_DURATION * 1_000_000,
            -1,
            str_to_char("Recording ends")
        );
        edfclose_file(hdl);
    }
}
