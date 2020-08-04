use anyhow::Error;
use kdtree::distance::squared_euclidean;
use kdtree::KdTree;
use ndarray::{array, Array1};
use ndarray_rand::{rand_distr::Uniform, RandomExt};
use plotters::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json as json;
use std::{fs::File, path::Path};

fn default_kdtree() -> KdTree<f64, (), [f64; 2]> {
    KdTree::new(2)
}

fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    let dist = squared_euclidean(a, b);
    dist.sqrt()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Class {
    Positive,
    Negative,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Model {
    pub detectors: Vec<Array1<f64>>,

    pub radius: f64,
    pub maximums: Array1<f64>,
    pub minimums: Array1<f64>,

    #[serde(skip)]
    #[serde(default = "default_kdtree")]
    pub kdtree: KdTree<f64, (), [f64; 2]>,
}

impl Model {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self {
            detectors: vec![],
            radius: 0.0,
            maximums: array![],
            minimums: array![],
            kdtree: default_kdtree(),
        }
    }

    pub fn save(&self, file: impl AsRef<Path>) -> Result<(), Error> {
        let file = File::create(file)?;
        Ok(json::to_writer_pretty(file, &self)?)
    }

    pub fn load(file: impl AsRef<Path>) -> Result<Self, Error> {
        let file = File::open(file)?;
        let mut model: Self = json::from_reader(file)?;

        model.init_kdtree()?;
        Ok(model)
    }

    pub fn init_kdtree(&mut self) -> Result<(), Error> {
        for detector in &self.detectors {
            self.kdtree.add([detector[0], detector[1]], ())?;
        }

        Ok(())
    }
}

pub struct FitParams<'a> {
    pub positives: &'a [Array1<f64>],
    pub radius: f64,
    pub maximums: &'a Array1<f64>,
    pub minimums: &'a Array1<f64>,
    pub no_detectors: usize,
}

pub struct PlotParams<'a, P: AsRef<Path>> {
    pub file: P,
    pub positives: &'a [Array1<f64>],
    pub maximums: &'a Array1<f64>,
    pub minimums: &'a Array1<f64>,
}

impl Model {
    pub fn fit(&mut self, params: &FitParams) -> Result<(), Error> {
        let mut temp_kdtree = default_kdtree();

        let normalized: Vec<_> = params
            .positives
            .iter()
            .map(|p| (p - params.minimums) / (params.maximums - params.minimums))
            .collect();

        for point in normalized {
            temp_kdtree.add([point[0], point[1]], ())?;
        }

        self.maximums = params.maximums.to_owned();
        self.minimums = params.minimums.to_owned();
        self.radius = params.radius;
        self.detectors = Vec::new();

        while self.detectors.len() < params.no_detectors {
            let detector = Array1::random((2,), Uniform::new(0.0, 1.0));
            let positives = temp_kdtree.within(
                &[detector[0], detector[1]],
                params.radius,
                &euclidean_distance,
            )?;

            if positives.is_empty() {
                println!("{}", detector);
                self.detectors.push(detector);
            }
        }

        self.init_kdtree()?;
        Ok(())
    }

    pub fn eval(&self, point: &Array1<f64>) -> Result<Class, Error> {
        println!("Punto recibido: {}", point);
        let normalized = (point - &self.minimums) / (&self.maximums - &self.minimums);

        println!("Punto normalizado: {}", normalized);

        let detectors = self.kdtree.within(
            &[normalized[0], normalized[1]],
            self.radius,
            &euclidean_distance,
        )?;

        Ok(if detectors.is_empty() {
            Class::Positive
        } else {
            Class::Negative
        })
    }

    pub fn test(&self, points: &[(Array1<f64>, Class)]) -> Result<f64, Error> {
        let total = points.len();
        let mut correct = 0;

        for (point, expected) in points {
            let predicted = self.eval(point)?;
            if *expected == predicted {
                println!("Predicción correcta!");
                correct += 1;
            } else {
                println!("Predicción incorrecta!");
            }
        }

        let precision = correct as f64 / total as f64;
        Ok(precision)
    }

    pub fn plot<P>(&self, params: &PlotParams<P>) -> Result<(), Error>
    where
        P: AsRef<Path>,
    {
        let root = BitMapBackend::new(&params.file, (1280, 720)).into_drawing_area();

        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .margin(10)
            .x_label_area_size(15)
            .y_label_area_size(15)
            .build_ranged(0.0..1.0, 0.0..1.0)?;

        chart
            .configure_mesh()
            .disable_x_mesh()
            .disable_y_mesh()
            .draw()?;

        chart.draw_series(
            self.detectors
                .iter()
                .map(|arr| Circle::new((arr[0], arr[1]), 2, ShapeStyle::from(&RED).filled())),
        )?;

        let normalized: Vec<_> = params
            .positives
            .iter()
            .map(|p| (p - params.minimums) / (params.maximums - params.minimums))
            .collect();

        chart.draw_series(
            normalized
                .iter()
                .map(|arr| Circle::new((arr[0], arr[1]), 2, ShapeStyle::from(&GREEN).filled())),
        )?;

        Ok(())
    }
}
