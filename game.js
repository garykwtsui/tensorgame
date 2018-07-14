let samples = '';
let main;
class Util {
    static isInDeadZone(x0, y0, x1, y1) {
        const margin = 150;
        const bubble = 10;
        let deadZone = new Phaser.Rectangle(x0 - margin, y0 - margin, margin * 2,
            margin * 2);
        let alienBubble = new Phaser.Rectangle(x1 - bubble, y1 - bubble, bubble * 2, bubble * 2);
        return Phaser.Rectangle.intersects(alienBubble, deadZone);
    }
    static downloadCSV(args) {
        let csv = 'data:text/csv;charset=utf-8,' + this.flattenForCSV(args.records);
        let data = encodeURI(csv);
        let link = document.createElement('a');
        link.setAttribute('href', data);
        link.setAttribute('download', args.filename);
        link.click();
    }
    static flattenForCSV(data) {
        console.log(data);
        let flattenedRecords = '';
        for (let i = 0; i < data.length; i++) {
            let row = '';
            for (let j = 0; j < data[i].aliens.length; j++) {
                for (let d = 0; d < data[i].aliens[j].length; d++) {
                    row += data[i].aliens[j][d] + ',';
                }
            }
            row += data[i].car[0] + ', ' + data[i].car[1]
            flattenedRecords += row + '\n';
        }

        return flattenedRecords;
    }
    static loadModel(evt) {
        let modelJSONFile;
        let weightsFile;
        if (evt.files.length > 0) {
            if (evt.files[0].name.indexOf('weight') || evt.files[0].name.indexOf('.bin')) {
                weightsFile = evt.files[0];
                modelJSONFile = evt.files[1];
            } else {
                weightsFile = evt.files[1];
                modelJSONFile = evt.files[0];
            }
            (async function (modelJSONFile, weightsFile) {
                const model = await tf.loadModel(tf.io.browserFiles([modelJSONFile, weightsFile]));
                if (model) {
                    console.log('Model: ' + model);
                    // do a brief test
                    const rawTestSample = [759.00, 3.00, 742.00, 62.00, 531.00, 538.00, 85.00, 175.00, 600.00, 189.00, 391.00,
                        90.00,
                        626.00, 401.00, 607.00, 350.00, 569.00, 109.00, 91.00, 101.00, 6.00, 406.00, 150.00, 250.00, 16.00, 42.00,
                        145.00, 98.00, 391.00, 106.00, 361.00, 67.00, 630.00, 100.00, 432.00, 47.00, 76.00, 407.00, 388.00, 75.00,
                        611.00, 390.00, 619.00, 131.00, 674.00, 174.00, 572.00, 282.00, 230.00, 599.00, 273.00, 137.00, 290.00,
                        588.00,
                        233.00, 487.00, 61.00, 355.00, 504.00, 102.00, 400.00, 62.00, 796.00, 327.00, 751.00, 494.00, 58.00, 278.00,
                        585.00, 362.00, 548.00, 469.00, 570.00, 529.00, 97.00, 393.00, 4.00, 340.00, 89.00, 151.00, 582.00, 415.00,
                        432.00, 130.00, 83.00, 22.00, 217.00, 255.00, 714.00, 122.00, 130.00, 343.00, 613.00, 170.00, 471.00, 3.00,
                        357.00, 8.00, 113.00, 345.00, 789.00, 305.00, 776.00, 56.00, 568.00, 15.00, 746.00, 351.00, 204.00, 79.00,
                        371.00, 40.00, 601.00, 294.00, 128.00, 33.00, 450.00, 579.00, 484.00, 25.00, 647.00, 319.00, 178.00, 323.00,
                        587.00, 444.00, 85.00, 265.00, 740.00, 54.00, 182.00, 263.00, 58.00, 387.00, 37.00, 592.00, 609.00, 457.00,
                        221.00, 251.00, 703.00, 133.00, 293.00, 36.00, 424.00, 38.00, 75.00, 524.00, 189.00, 437.00, 726.00, 292.00,
                        19.00, 321.00, 96.00, 226.00, 752.00, 426.00, 127.00, 443.00, 775.00, 300.00, 68.00, 494.00, 436.00, 576.00,
                        196.00, 397.00, 218.00, 345.00, 484.00, 589.00, 707.00, 532.00, 399.00, 90.00, 584.00, 95.00, 660.00, 35.00,
                        680.00, 128.00, 731.00, 209.00, 704.00, 473.00, 636.00, 306.00, 448.00, 114.00, 560.00, 88.00, 738.00, 147.00,
                        383.00, 88.00, 46.00, 477.00, 321.00, 110.00
                    ];
                    let testSample = [];
                    while (rawTestSample.length) {
                        // need to apply normalization
                        let values = rawTestSample.splice(0, 2);
                        values[0] = values[0] / 800;
                        values[1] = values[1] / 600;
                        testSample.push(values);
                    }
                    const sample = tf.reshape(tf.tensor(testSample), [1, 100, 2]);
                    const result = model.predict(sample).dataSync();
                    // TODO: perhaps perform other ways to check
                    if (result != undefined) {
                        console.log('Model is loaded successfully');
                        main = new MainApp(document.getElementById('gamePanel'), model);
                        main.start();
                    }
                } else {
                    alert('Unable to load model');
                }

            })(modelJSONFile, weightsFile);
        } else {
            alert('No model information provided..');
        }
    }
}

class Game {
    constructor(elem, model) {
        this.trainedModel = model;
        this.gameDimension = {
            width: 800,
            height: 600
        };
        this.panelElement = elem;
        this.aliens;
        this.car;
        this.cursor;
        this.spaceKey;
        this.game;
        this.history = [];
        this.historyReplayCounter = -1;
        this.carStartPos = {
            x: this.gameDimension.width / 2,
            y: this.gameDimension.height / 2
        };
        this.screenDrawingRefs = [];
    }
    start() {
        this.game = new Phaser.Game(this.gameDimension.width, this.gameDimension.height, Phaser.CANVAS,
            this.panelElement, {
                preload: this.preload.bind(this),
                create: this.create.bind(this),
                update: this.update.bind(this),
                render: this.render
            });
    }

    preload() {
        this.game.load.image('car', './assets/sprites/car90.png');
        this.game.load.image('baddie', './assets/sprites/space-baddie.png');
    }

    create() {
        this.game.physics.startSystem(Phaser.Physics.ARCADE);

        this.aliens = this.game.add.group();
        this.aliens.enableBody = true;

        let i = 0;
        while (i < 100) {
            let randomPos = {
                x: this.game.world.randomX,
                y: this.game.world.randomY
            }
            if (!Util.isInDeadZone(this.carStartPos.x, this.carStartPos.y, randomPos.x, randomPos.y)) {
                let s = this.aliens.create(randomPos.x, randomPos.y, 'baddie');
                s.name = 'alien' + s;
                s.body.collideWorldBounds = true;
                s.body.bounce.setTo(0.8, 0.8);
                s.body.velocity.setTo(10 + Math.random() * 40, 10 + Math.random() * 40);
                i++;
            }
        }

        this.car = this.game.add.sprite(this.carStartPos.x, this.carStartPos.y, 'car');
        this.car.name = 'car';
        this.car.anchor.set(0.5);

        this.game.physics.enable(this.car, Phaser.Physics.ARCADE);

        this.car.body.collideWorldBounds = true;
        this.car.body.bounce.set(0.8);
        this.car.body.allowRotation = true;
        this.car.body.immovable = true;

        this.cursors = this.game.input.keyboard.createCursorKeys();
        // this.spaceKey = this.game.input.keyboard.addKey(Phaser.Keyboard.SPACEBAR);
        // this.spaceKey.onDown.add(this.togglePause, this.game);
    }

    togglePause() {
        this.game.physics.arcade.isPaused = (this.game.physics.arcade.isPaused) ? false : true;
    }

    update() {
        if (this.historyReplayCounter >= 0) {
            if (this.historyReplayCounter >= this.history.length) {
                this.game.physics.arcade.isPaused = true;
                // this.historyReplayCounter = -1;
                this.showEndGameText();
                return;
            }
            let state = this.history[this.historyReplayCounter++];
            for (let k = 0; k < state.aliens.length; k++) {
                this.aliens.children[k].position.x = state.aliens[k][0] * this.gameDimension.width;
                this.aliens.children[k].position.y = state.aliens[k][1] * this.gameDimension.height;
            }
            this.car.position.x = state.car[0] * this.gameDimension.width;
            this.car.position.y = state.car[1] * this.gameDimension.height;
        } else {
            let hit = this.game.physics.arcade.collide(this.car, this.aliens);
            if (hit) {
                this.game.physics.arcade.isPaused = true;
                Util.downloadCSV({
                    filename: "game-data.csv",
                    records: this.history
                });
                // alert('Crashed!');
                this.showEndGameText();
            } else {
                this.car.body.velocity.x = 0;
                this.car.body.velocity.y = 0;
                this.car.body.angularVelocity = 0;

                if (this.cursors.left.isDown) {
                    this.car.body.angularVelocity = -200;
                } else if (this.cursors.right.isDown) {
                    this.car.body.angularVelocity = 200;
                }

                if (this.cursors.up.isDown) {
                    this.car.body.velocity.copyFrom(this.game.physics.arcade.velocityFromAngle(this.car.angle, 300));
                }

                if (this.trainedModel) {
                    let positions = this.getCurrentPositions(false);
                    let currentState = positions.aliens;
                    let sample = tf.reshape(tf.tensor(currentState), [1, 100, 2]);
                    const result = this.trainedModel.predict(sample).dataSync();
                    this.car.position.x = result[0] * this.gameDimension.width;
                    this.car.position.y = result[1] * this.gameDimension.height;
                }

                // for replay purposes
                this.history.push(this.getCurrentPositions(true));

            }
        }
    }

    getCurrentPositions(includeCarPosition) {
        let aliens = [];
        let car = [];
        for (let i = 0; i < this.aliens.children.length; i++) {
            let position = [this.aliens.children[i].position.x.toFixed(2) / this.gameDimension.width, this.aliens.children[i]
                .position.y.toFixed(
                    2) / this.gameDimension.height
            ];
            aliens.push(position);
        }
        if (includeCarPosition) {
            car = [this.car.position.x.toFixed(2) / this.gameDimension.width, this.car.position.y.toFixed(2) / this.gameDimension
                .height
            ];
        }
        return {
            aliens: aliens,
            car: car
        }
    }

    render() {}

    replay() {
        this.game.physics.arcade.isPaused = false;
        this.historyReplayCounter = 0;
    }

    showEndGameText() {
        // let bar = this.game.add.graphics();
        // bar.beginFill(0x000000, 0.2);
        // bar.drawRect(this.gameDimension.width / 2, this.gameDimension.height / 2, 800, 100);

        var style = {
            font: "bold 32px Arial",
            fill: "#fff",
            boundsAlignH: "center",
            boundsAlignV: "middle"
        };

        //  The Text is positioned at 0, 100
        let text = this.game.add.text(0, 0, "Crashed", style);
        text.setShadow(3, 3, 'rgba(0,0,0,0.5)', 2);

        //  We'll set the bounds to be from x0, y100 and be 800px wide by 100px high
        text.setTextBounds(0, 100, 800, 100);

        this.screenDrawingRefs.push(text);
    }
    clearScreen() {
        for (let i = 0; i < this.screenDrawingRefs.length; i++) {
            this.screenDrawingRefs[i].destroy();
        }
    }
}

class MainApp {
    constructor(elem, model) {
        this.game = new Game(elem, model);
    }
    start() {
        this.game.start();
    }
    replay() {
        this.game.clearScreen();
        this.game.replay();
    }
}

document.getElementById('button-open-gamepanel').addEventListener('click', () => {
    main = new MainApp(document.getElementById('gamePanel'));
    main.start();
});

document.getElementById('button-replay-game').addEventListener('click', () => {
    main.replay();
});

document.getElementById('model-file-selector').addEventListener('change', () => {
    Util.loadModel(document.getElementById('model-file-selector'));
});