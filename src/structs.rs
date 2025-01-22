pub(crate) struct SourceWithLimits<T> {
    pub source: T,
    pub min_zoom: u32,
    pub max_zoom: u32,
    pub min_x: u32,
    pub max_x: u32,
    pub min_y: u32,
    pub max_y: u32,
}
