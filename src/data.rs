use anyhow::Error;
use serde::Deserialize;
use serde_json as json;
use std::{fs::File, path::Path};

#[derive(Debug, Clone, Deserialize)]
pub struct Range {
    pub max: f64,
    pub min: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Metadata {
    pub sepal_length: Range,
    pub sepal_width: Range,
    pub petal_length: Range,
    pub petal_width: Range,
}

impl Metadata {
    pub fn load_json(file: impl AsRef<Path>) -> Result<Self, Error> {
        Ok(json::from_reader(File::open(file)?)?)
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Record {
    pub sepal_length: f64,
    pub sepal_width: f64,
    pub petal_length: f64,
    pub petal_width: f64,
    pub class: String,
}

impl Record {
    pub fn load_csv(file: impl AsRef<Path>) -> Result<Vec<Self>, Error> {
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(true)
            .trim(csv::Trim::All)
            .from_path(file)?;

        let records: Result<Vec<Self>, _> = reader.deserialize().collect();
        Ok(records?)
    }
}
