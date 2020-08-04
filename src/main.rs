pub mod algo;
pub mod data;

use algo::{Class, FitParams, Model, PlotParams};
use anyhow::Error;
use ndarray::array;

fn main() -> Result<(), Error> {
    let metadata = data::Metadata::load_json("data/metadata.json")?;

    println!("*** Parámetros ***");
    println!("Dos clases: Iris-setosa (+)  e Iris-versicolor (-)");
    println!("Dos atributos:  sepal_length, petal_length");
    println!("Radio (r): 0.05");
    println!("# de detectores: 1000");
    println!("max(sepal_length): {}", metadata.sepal_length.max);
    println!("min(sepal_length): {}", metadata.sepal_length.min);
    println!("max(petal_length): {}", metadata.petal_length.max);
    println!("min(petal_length): {}", metadata.petal_length.min);

    // Load only 'positive' points, class 'Iris-setosa'
    let train_data: Vec<_> = data::Record::load_csv("data/train.csv")?
        .into_iter()
        .filter_map(|record| {
            if record.class == "Iris-setosa" {
                Some(array![record.sepal_length, record.petal_length])
            } else {
                None
            }
        })
        .collect();

    println!();
    println!("*** Self (positivos) ***");
    for positive in &train_data {
        println!("{}", positive);
    }

    let maximums = array![metadata.sepal_length.max, metadata.petal_length.max];
    let minimums = array![metadata.sepal_length.min, metadata.petal_length.min];

    println!();
    println!("*** Detectores ***");
    let mut model = Model::new();
    model.fit(&FitParams {
        positives: &train_data,
        radius: 0.05,
        maximums: &maximums,
        minimums: &minimums,
        no_detectors: 1000,
    })?;

    // Load all the points, class 'Iris-setosa' and 'Iris-versicolor'
    let test_data: Vec<_> = data::Record::load_csv("data/test.csv")?
        .into_iter()
        .filter_map(|record| match record.class.as_str() {
            "Iris-setosa" => Some((
                array![record.sepal_length, record.petal_length],
                Class::Positive,
            )),
            "Iris-versicolor" => Some((
                array![record.sepal_length, record.petal_length],
                Class::Negative,
            )),
            _ => None,
        })
        .collect();

    println!("*** Data de prueba ***");
    for (point, class) in &test_data {
        let class_msg = if let Class::Positive = class {
            "positivo"
        } else {
            "negativo"
        };

        println!("Clase: {}, {}", class_msg, point);
    }

    println!();
    println!("Evaluando puntos");
    let precision = model.test(&test_data)?;
    println!("Total: {} muestras", test_data.len());
    println!("Precisión: {}", precision);

    model.plot(&PlotParams {
        file: "result.png",
        positives: &train_data,
        maximums: &maximums,
        minimums: &minimums,
    })?;

    Ok(())
}
