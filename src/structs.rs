use std::{collections::HashMap, path::PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct SourceLimits {
    pub min_x: u32,
    pub max_x: u32,
    pub min_y: u32,
    pub max_y: u32,
}

pub struct SourceWithLimits {
    pub source: PathBuf,
    pub limits: HashMap<u8, SourceLimits>,
}

pub struct TileData {
    /// 1 2<br>
    /// 3 4
    pub quad: u8,
    pub rgb: Vec<u8>,
    pub alpha: Vec<u8>,
}
