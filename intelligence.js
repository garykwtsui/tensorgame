import * as tf from '@tensorflow/tfjs';
import fs from 'fs';
var math = require('mathjs');
const parse = require('csv-parse/lib/sync')

class Config {}
Config.epochs = 50;
// Config.window_size = 50;
Config.multiplier = 1.00;
Config.hidden_layer_size = [
    1024 * Config.multiplier,
    512 * Config.multiplier,
    // 256 * Config.multiplier,
    // 128 * Config.multiplier,
    // 64 * Config.multiplier,
    // 32 * Config.multiplier,
    // 16 * Config.multiplier
];


class DataFeed {
    constructor(csvPath) {
        this.csvPath = csvPath;
    }

    loadData(delimiter, postProcessFunc) {
        console.log("Going to read: " + this.csvPath);
        // TODO:    looks like fs.readFileSync reads a string at 'build-time' instead of run-time, so passing a param in
        //          will not work
        const data = fs.readFileSync('game-data.csv', 'utf8');
        if (!delimiter) {
            return this.parseData(data);
        } else {
            return data.split(delimiter);
        }
    }

    parseData(data) {
        if (!data) {
            throw "no data";
        }
        var records = parse(data, {
            delimiter: ',',
            // columns: true,
            // cast: true
        });

        let results = [];
        for (let i = 0; i < records.length; i++) {
            let row = records[i];
            let result = [];
            while (row.length) {
                let sample = row.splice(0, 2);
                sample[0] = sample[0] / 800;
                sample[1] = sample[1] / 600;

                // result.push(row.splice(0, 2));
                result.push(sample);
            }
            results.push(result);
        }
        console.log("Records read: " + results.length);

        return results;
    }
}

class DataFormatter {
    constructor(data) {
        this.data = data;
    }

    stride() {
        var X = [];
        var Y = [];

        for (let i = 0; i < this.data.length; i++) {
            const row = this.data[i].slice(0, this.data[i].length - 1);
            const label = this.data[i].slice(this.data[i].length - 1, this.data[i].length);
            X.push(row);
            Y.push(label);
        }

        return {
            training: X,
            label: Y
        };
    }
}

class Intelligence {
    constructor(x_train, y_train, x_test, y_test) {
        this.x_train = x_train;
        this.y_train = y_train;
        this.x_test = x_test;
        this.y_test = y_test;
    }

    buildModel() {
        console.log('building with a NN');
        const model = tf.sequential();

        model.add(tf.layers.dense({
            units: Config.hidden_layer_size["1"],
            inputShape: [100, 2],
            activation: 'relu'
        }));

        for (let i = 0; i < Config.hidden_layer_size.length; i++) {
            model.add(tf.layers.dense({
                units: Config.hidden_layer_size[i],
            }));
        }

        model.add(tf.layers.flatten());
        model.add(tf.layers.dense({
            units: 2,
            activation: 'linear'
        }));

        // const optimizer = tf.train.adam(0.2);
        model.compile({
            loss: 'meanSquaredError',
            optimizer: 'rmsprop',
            // optimizer: optimizer,
            metrics: ['accuracy']
        });

        return model;
    }

    // TODO: work out below.
    buildModelWithRnn() {
        console.log('building with a simple RNN');
        const model = tf.sequential();
        model.add(tf.layers.simpleRNN({
            units: Config.hidden_layer_size,
            recurrentInitializer: 'glorotNormal',
            inputShape: [100, 2]
        }));
        const N = 1;
        model.add(tf.layers.repeatVector({
            n: N
        }));
        model.add(tf.layers.simpleRNN({
            units: Config.hidden_layer_size,
            recurrentInitializer: 'glorotNormal',
            returnSequences: true
        }));
        model.add(tf.layers.timeDistributed({
            layer: tf.layers.dense({
                units: 2,
            })
        }));
        model.add(tf.layers.activation({
            activation: 'softmax',
        }));
        model.compile({
            loss: 'meanSquaredError',
            optimizer: 'adam',
            metrics: ['accuracy']
        });

        return model;

    }

    async train() {
        // setup model
        const model = this.buildModel();
        // const model = this.buildModelWithRnn();

        let samples = tf.reshape(tf.tensor(this.x_train), [this.x_train.length, 100, 2]);
        let labels = tf.reshape(tf.tensor(this.y_train), [this.y_train.length, 2]);
        // let test_data = tf.reshape(tf.tensor2d(this.x_test), [this.x_test.length, Config.window_size, 1]);
        // let test_labels = tf.reshape(tf.tensor1d(this.y_test), [this.y_test.length, 1]);

        console.log("Going to train now");
        const history = await model.fit(samples, labels, {
            epochs: Config.epochs,
            shuffle: false,
            validationSplit: 0.05,
            callbacks: {
                onEpochBegin: async (epoch, log) => {
                    console.log(`Epoch ${epoch} started.`);
                },
                onEpochEnd: async (epoch, log) => {
                    console.log(`Epoch ${epoch}: loss = ${log.loss}`);
                }
            }
        });
        this.outputModelAccuracy(history);
        return model;
    }

    outputModelAccuracy(history) {
        const trainLoss = history.history['loss'][0];
        const trainAccuracy = history.history['acc'][0];
        const valLoss = history.history['val_loss'][0];
        const valAccuracy = history.history['val_acc'][0];
        console.log('Train Loss: ' + trainLoss);
        console.log('Train Acc: ' + trainAccuracy);
        console.log('Val Loss: ' + valLoss);
        console.log('Val Acc: ' + valAccuracy);
    }
}

class MainDriver {
    static main() {
        // 0. loading data
        console.log("Loading data");
        const dataFeed = new DataFeed('game-data.csv');
        const data = dataFeed.loadData();
        const windowSize = data[0].length - 1;
        // 1. striding
        console.log("Preprocessing data");
        const results = new DataFormatter(data).stride();
        const length = results.training.length;

        // 2. prepping data sets
        console.log("Preparing data sets");
        const trainingSize = length;
        const x_train = results.training.slice(0, trainingSize);
        // const x_test = results.training.slice(length - (length * 0.1), length);
        const y_train = results.label.slice(0, trainingSize);
        // const y_test = results.label.slice(length - (length * 0.1), length);

        // 3. training and predicting
        const intelli = new Intelligence(x_train, y_train);
        (async function (intelli) {
            console.log("Going to train now -- may take a while");
            const model = await intelli.train();

            console.log("Going to predict now");
            //predicting all on a point by point basis.
            var outputStr = '';
            for (let k = 0; k < 1; k++) {
                const test = results.training.slice(0, trainingSize);
                const sample = tf.reshape(tf.tensor(test[k]), [1, 100, 2]);
                const label = results.label[k];
                const result = model.predict(sample);
                outputStr += result.dataSync() + ', ' + label + '\n';
            }

            console.log("Outputting results -- may take a while");
            console.log(outputStr);

            console.log('Going to save now');
            const saveResults = await model.save('downloads://my-model-1');
            console.log('done');
        })(intelli);
    }
}

document.getElementById('mainButton-gamedata').addEventListener('click', () => {
    MainDriver.main();
});