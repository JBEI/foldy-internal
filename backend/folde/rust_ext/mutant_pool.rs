use std::collections::HashSet;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;

#[derive(Clone, Debug, Eq, PartialEq)]
enum ParsedSeqId {
    WildType,
    Homolog,
    Alleles(Vec<String>),
}

fn parse_seq_id(seq_id: &str) -> ParsedSeqId {
    if seq_id == "WT" {
        return ParsedSeqId::WildType;
    }
    if seq_id.starts_with("HOM-") {
        return ParsedSeqId::Homolog;
    }
    ParsedSeqId::Alleles(seq_id.split('_').map(str::to_string).collect())
}

fn symmetric_difference_len(left: &[String], right: &[String]) -> usize {
    let mut distance = 0usize;
    for allele in left {
        if !right.contains(allele) {
            distance += 1;
        }
    }
    for allele in right {
        if !left.contains(allele) {
            distance += 1;
        }
    }
    distance
}

fn get_mutant_pool(seq_ids: &[&str], measured_seq_ids: &[&str]) -> Vec<String> {
    let measured_raw: HashSet<&str> = measured_seq_ids.iter().copied().collect();
    let measured_allele_sets: Vec<Vec<String>> = measured_seq_ids
        .iter()
        .filter_map(|seq_id| match parse_seq_id(seq_id) {
            ParsedSeqId::Alleles(alleles) => Some(alleles),
            ParsedSeqId::WildType => Some(Vec::new()),
            ParsedSeqId::Homolog => None,
        })
        .collect();

    let mut mutant_pool = Vec::new();
    for seq_id in seq_ids {
        if measured_raw.contains(seq_id) {
            continue;
        }

        match parse_seq_id(seq_id) {
            ParsedSeqId::Homolog => {
                mutant_pool.push((*seq_id).to_string());
            }
            ParsedSeqId::WildType => {
                if measured_allele_sets
                    .iter()
                    .any(|measured| symmetric_difference_len(&[], measured) == 1)
                {
                    mutant_pool.push((*seq_id).to_string());
                }
            }
            ParsedSeqId::Alleles(alleles) => {
                if alleles.len() == 1 {
                    mutant_pool.push((*seq_id).to_string());
                    continue;
                }
                if measured_allele_sets
                    .iter()
                    .any(|measured| symmetric_difference_len(&alleles, measured) == 1)
                {
                    mutant_pool.push((*seq_id).to_string());
                }
            }
        }
    }
    mutant_pool
}

fn split_lines(input: &str) -> Vec<&str> {
    input.lines().filter(|line| !line.is_empty()).collect()
}

#[no_mangle]
pub extern "C" fn foldy_get_mutant_pool(
    seq_ids_ptr: *const c_char,
    measured_seq_ids_ptr: *const c_char,
) -> *mut c_char {
    if seq_ids_ptr.is_null() || measured_seq_ids_ptr.is_null() {
        return std::ptr::null_mut();
    }

    let seq_ids_input = unsafe { CStr::from_ptr(seq_ids_ptr) };
    let measured_seq_ids_input = unsafe { CStr::from_ptr(measured_seq_ids_ptr) };
    let Ok(seq_ids_str) = seq_ids_input.to_str() else {
        return std::ptr::null_mut();
    };
    let Ok(measured_seq_ids_str) = measured_seq_ids_input.to_str() else {
        return std::ptr::null_mut();
    };

    let seq_ids = split_lines(seq_ids_str);
    let measured_seq_ids = split_lines(measured_seq_ids_str);
    let result = get_mutant_pool(&seq_ids, &measured_seq_ids).join("\n");

    match CString::new(result) {
        Ok(c_string) => c_string.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn foldy_free_string(ptr: *mut c_char) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        let _ = CString::from_raw(ptr);
    }
}
