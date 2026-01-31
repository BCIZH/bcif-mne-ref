use std::{ ffi::{ CStr, CString }, os::raw::c_char };

pub fn str_to_char(input: &str) -> *const c_char {
    CString::new(input).unwrap().into_raw()
}

pub fn char_to_str(ptr: *mut i8) -> String {
    let cstr = unsafe { CStr::from_ptr(ptr) };
    let result = cstr.to_str().unwrap().to_owned().to_string();
    result
}
