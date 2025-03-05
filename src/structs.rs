use std::{collections::HashMap, path::PathBuf};

use serde::{Deserialize, Serialize};

use crate::background::Background;

#[derive(Serialize, Deserialize)]
pub struct SourceLimits {
    pub min_x: u32,
    pub max_x: u32,
    pub min_y: u32,
    pub max_y: u32,
}

pub struct SourceWithLimits {
    pub source: PathBuf,
    pub limits: Option<HashMap<u8, SourceLimits>>,
}

pub struct TileShift {
    pub x: u8,
    pub y: u8,
    pub level: u8,
}

pub struct TileData {
    pub rgb: Vec<u8>,
    pub alpha: Vec<u8>,
    pub shift: Option<TileShift>,
}

pub struct Context {
    pub sources: Vec<SourceWithLimits>,
    pub default_background: Background,
    pub verbosity: u8,
}
