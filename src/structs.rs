use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct SourceLimits {
    pub min_x: u32,
    pub max_x: u32,
    pub min_y: u32,
    pub max_y: u32,
}
