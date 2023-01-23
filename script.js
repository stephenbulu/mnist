
import {TrainingData} from './data.js';


async function showExamples(data) {
  const surface =
    tfvis.visor().surface({ name: 'Input Data Examples', tab: 'Input Data'});  
  const examples = data.nextTestBatch(20);
  const numExamples = examples.xs.shape[0];
  for (let i = 0; i < numExamples; i++) {
    const imageTensor = tf.tidy(() => {
      return examples.xs
        .slice([i, 0], [1, examples.xs.shape[1]])
        .reshape([28, 28, 1]);
    });
    
    const canvas = document.createElement('canvas');
    canvas.width = 28;
    canvas.height = 28;
    canvas.style = 'margin: 4px;';
    await tf.browser.toPixels(imageTensor, canvas);
    surface.drawArea.appendChild(canvas);

    imageTensor.dispose();
  }
}

function getModel() {
    const model = tf.sequential();
    
    const IMAGE_WIDTH = 28;
    const IMAGE_HEIGHT = 28;
    const IMAGE_CHANNELS = 1;  
    
    // model.add(tf.layers.flatten({
    //   inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS]
    // }));

    // model.add(tf.layers.dense({
    //   units:10,
    //   activation:'sigmoid'
    // }))


    model.add(tf.layers.conv2d({
      inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],
      kernelSize: 5,
      filters: 8,
      strides: 1,
      activation: 'relu',
      kernelInitializer: 'varianceScaling',
    }));
  

    model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));

    model.add(tf.layers.conv2d({
      kernelSize: 5,
      filters: 16,
      strides: 1,
      activation: 'relu',
      kernelInitializer: 'varianceScaling'
    }));
    model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
    

    model.add(tf.layers.flatten());

    const NUM_OUTPUT_CLASSES = 10;
    model.add(tf.layers.dense({
      units: NUM_OUTPUT_CLASSES,
      kernelInitializer: 'varianceScaling',
      activation: 'softmax'
    }));
  

    const optimizer = tf.train.adam();
    model.compile({
      optimizer: optimizer,
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy'],
    });
  
    return model;
}

async function train(model, data) {
    const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
    const container = {
      name: 'Model Training', tab: 'Model', styles: { height: '1000px' }
    };
    const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);
    
    const BATCH_SIZE = 512;
    const TRAIN_DATA_SIZE = 55000;
    const TEST_DATA_SIZE = 10000;
  
    const [trainXs, trainYs] = tf.tidy(() => {
      const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
      return [
        d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]),
        d.labels
      ];
    });
  
    const [testXs, testYs] = tf.tidy(() => {
      const d = data.nextTestBatch(TEST_DATA_SIZE);
      return [
        d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]),
        d.labels
      ];
    });
  
    return model.fit(trainXs, trainYs, {
      batchSize: BATCH_SIZE,
      validationData: [testXs, testYs],
      epochs: 10,
      shuffle: true,
      callbacks: fitCallbacks
    });
}

var model;
async function startUp() {
    // create canvas element and append it to document body
    var canvas = document.createElement('canvas');
    canvas.id = "drawCanvas"
    document.body.appendChild(canvas);
    // document.body.style.textAlign = "center"
    // some hotfixes... ( ≖_≖)
    document.body.style.margin = 0;
    //canvas.style.position = 'fixed';

    // get canvas 2D context and set him correct size
    var ctx = canvas.getContext('2d',{ willReadFrequently: true } );

    resize();

    // last known position
    var pos = { x: 0, y: 0 };


    canvas.addEventListener('mousemove', draw);
    document.addEventListener('mousedown', setPosition);
    document.addEventListener('mouseenter', setPosition);

    // new position from mouse event
    function setPosition(e) {
    const rect = canvas.getBoundingClientRect()
    // const x = event.clientX - rect.left
    // const y = event.clientY - rect.top
    pos.x = e.clientX - rect.left
    pos.y = e.clientY - rect.top
    }

    // resize canvas
    function resize() {
    ctx.canvas.width = 28;
    ctx.canvas.height = 28;
    }
    const canvasWidth = 300;
    const canvasHeight = 300;
    const ctxWidth = 28;
    const ctxHeight = 28;

    function draw(e) {
        // mouse left button must be pressed
        if (e.buttons !== 1) return;

        ctx.beginPath(); // begin

        ctx.lineWidth = 2;
        ctx.lineCap = 'square';
        ctx.strokeStyle = '#FFFFFF';

        ctx.moveTo(pos.x*(ctxWidth/canvasWidth), pos.y*(ctxHeight/canvasHeight)); // from
        setPosition(e);
        ctx.lineTo(pos.x*(ctxWidth/canvasWidth), pos.y*(ctxHeight/canvasHeight)); // to

        ctx.stroke(); // draw it!
        var imageData = ctx.getImageData(0, 0, 28, 28);
        for(var x=0; x<imageData.width; x++){
          for(var y=0; y<imageData.height; y++){
            var dataI = (x*imageData.width + y)*4
            // imageData.data[dataI+0] = Math.ceil( imageData.data[dataI+0]/255 - 0.5 ) * 255
            // imageData.data[dataI+1] = Math.ceil( imageData.data[dataI+1]/255 - 0.5 ) * 255
            // imageData.data[dataI+2] = Math.ceil( imageData.data[dataI+2]/255 - 0.5 ) * 255
            imageData.data[dataI+3] = Math.ceil( imageData.data[dataI+2]/255 - 0.5 ) * 255
          }
        }
        ctx.putImageData(imageData, 0, 0);
        updateCanvas()
    }
    // var btn = document.createElement('button');

    // btn.innerHTML = "Click Me";
    // btn.addEventListener('click', buttonClicked);
    // document.body.appendChild(btn);
    function updateCanvas(){
        const IMAGE_SIZE = 784;

        const datasetBytesView = new Float32Array(IMAGE_SIZE);

        var imageData = ctx.getImageData(0, 0, 28, 28);

        for (let j = 0; j < imageData.data.length / 4; j++) {
            datasetBytesView[j] = imageData.data[j * 4] / 255;
          }
        const xs = tf.tensor2d(datasetBytesView, [1, IMAGE_SIZE]);
        const bla = xs.reshape([1, 28, 28, 1]);
        //const preds = model.predict(bla);
        //const func = tf.function([model.layers[0].input], model.layers[1].output)
        const featExtractionModel = tf.model({
          inputs: model.input,
          outputs: model.layers.map(layer => layer.output)
        });
        var layersOutput = featExtractionModel.predict(bla).map(x => x.arraySync() )
        document.getElementById("pred").innerHTML = (layersOutput[5][0].indexOf(Math.max(...layersOutput[5][0])))
        

        const layer1output = document.getElementById("layer1output")
        for(i=0;i<8;i++){
          const oCtx = layer1output.children[i].getContext("2d");
          imageData = oCtx.createImageData(24, 24);

          var leastVal = 0
          var greatestVal = 0
          
          for(var x = 0;x<24;x++){
            for(var y=0;y<24;y++){
              greatestVal = Math.max(Math.abs(layersOutput[0][0][x][y][i]), greatestVal)
              leastVal = Math.min(layersOutput[0][0][x][y][i], leastVal)

            }
          }
          var diffVal = greatestVal-leastVal
          for(var x = 0;x<24;x++){
            for(var y=0;y<24;y++){
              //var v = (diffVal - (greatestVal - layersOutput[0][0][x][y][i])) / diffVal * 255
              var v = layersOutput[0][0][x][y][i] * ( 255/greatestVal)
              var r = v
              if(v < 0){
                r = v*-1;
              }
              imageData.data[(x*24 + y)*4+0] = r
              imageData.data[(x*24 + y)*4+1] = v
              imageData.data[(x*24 + y)*4+2] = v
              imageData.data[(x*24 + y)*4+3] = 255
            }
          }
          oCtx.putImageData(imageData, 0, 0)
        }

        const layer2output = document.getElementById("layer2output")
        for(i=0;i<16;i++){
          const oCtx = layer2output.children[i].getContext("2d");
          imageData = oCtx.createImageData(8, 8);

          var leastVal = 0
          var greatestVal = 0
          for(var x = 0;x<8;x++){
            for(var y=0;y<8;y++){
              greatestVal = Math.max(layersOutput[2][0][x][y][i], greatestVal)
              leastVal = Math.min(layersOutput[2][0][x][y][i], leastVal)

            }
          }
          var diffVal = greatestVal-leastVal
          for(var x = 0;x<8;x++){
            for(var y=0;y<8;y++){
              var v = (diffVal - (greatestVal - layersOutput[2][0][x][y][i])) / diffVal * 255
              imageData.data[(x*8 + y)*4+0] = v
              imageData.data[(x*8 + y)*4+1] = v
              imageData.data[(x*8 + y)*4+2] = v
              imageData.data[(x*8 + y)*4+3] = 255
            }
          }
          oCtx.putImageData(imageData, 0, 0)
        }



        const layer3output = document.getElementById("layer3output")
        for(i=0;i<16;i++){
          const oCtx = layer3output.children[i].getContext("2d");
          imageData = oCtx.createImageData(8, 8);

          var leastVal = 0
          var greatestVal = 0
          for(var x = 0;x<4;x++){
            for(var y=0;y<4;y++){
              greatestVal = Math.max(layersOutput[3][0][x][y][i], greatestVal)
              leastVal = Math.min(layersOutput[3][0][x][y][i], leastVal)

            }
          }
          var diffVal = greatestVal-leastVal
          for(var x = 0;x<4;x++){
            for(var y=0;y<4;y++){
              var v = (diffVal - (greatestVal - layersOutput[3][0][x][y][i])) / diffVal * 255
              imageData.data[(x*8 + y)*4+0] = v
              imageData.data[(x*8 + y)*4+1] = v
              imageData.data[(x*8 + y)*4+2] = v
              imageData.data[(x*8 + y)*4+3] = 255
            }
          }
          oCtx.putImageData(imageData, 0, 0)
        }


    }
    ///// draw all of the default canvasas
    var predContainer = document.createElement('div');
    predContainer.id = "predContainer";
    var pred = document.createElement('H1');
    predContainer.innerHTML = "Prediction: ";
    pred.innerHTML = "-";
    pred.id = "pred";
    predContainer.appendChild(pred);
    document.body.appendChild(predContainer);
    document.body.appendChild(document.createElement("br"));
    var clearBtn = document.createElement('button');
    clearBtn.style.fontSize = '20px'
    //clearBtn.style.top = '30px'
    clearBtn.innerHTML = "Clear Canvas";
    //clearBtn.style.position = "fixed"
    function clearButtonClicked(){
        ctx.clearRect(0, 0, 28, 28);
    }
    clearBtn.addEventListener('click', clearButtonClicked);
    document.body.appendChild(clearBtn);

    document.body.appendChild(document.createElement("br"));

    const layer1 = document.createElement("div");
    layer1.id = "layer1"
    const layer1Features = model.layers[0].weights[0].read().arraySync();
    const layer1Bias = model.layers[0].weights[1].read().arraySync();
    console.log("layer 1,features", layer1Features, "bias", layer1Bias)

    for(var i =0;i<8;i++){
      const newBox = document.createElement("canvas");
      
      newBox.width = 5;
      newBox.height = 5;
      const newBoxCtx = newBox.getContext("2d");
      var imageData = newBoxCtx.createImageData(5, 5);

      newBoxCtx.fillStyle = "green";
      newBoxCtx.fillRect(0, 0, 5, 5);

      var leastVal = 0
      var greatestVal = 0
      for(var x = 0;x<5;x++){
        for(var y=0;y<5;y++){
          var weight = layer1Features[x][y][0][i];
          var bias = layer1Bias[i];
          greatestVal = Math.max(
            Math.abs( 255*(weight+bias)), greatestVal
          )

        }
      }
      for(var x = 0;x<5;x++){
        for(var y=0;y<5;y++){
          var weight = layer1Features[x][y][0][i];
          var bias = layer1Bias[i];
          //var v = ( weight+bias) / greatestVal * (254/2)
          var v = 255*(weight+bias) * (255/greatestVal)
          var r = v;
          if( v < 1){
            r = v*-1;
          }
          imageData.data[(x*5 + y)*4+0] = r
          imageData.data[(x*5 + y)*4+1] = v
          imageData.data[(x*5 + y)*4+2] = v
          imageData.data[(x*5 + y)*4+3] = 255
        }
      }
      newBoxCtx.putImageData(imageData, 0, 0);
      newBox.className = "featureCanvas";
      layer1.appendChild(newBox);
    }
    document.body.appendChild(layer1);
    document.body.appendChild(document.createElement("br"));
    const layer1output = document.createElement("div");
    layer1output.id = "layer1output"
    for(var i =0;i<8;i++){
      const newBox = document.createElement("canvas");
      
      newBox.width = 24;
      newBox.height = 24;
      const newBoxCtx = newBox.getContext("2d");

      newBoxCtx.fillStyle = "black";
      newBoxCtx.fillRect(0, 0, 24, 24);


      newBox.className = "layer1outputCanvas";
      layer1output.appendChild(newBox);
    }
    document.body.appendChild(layer1output);



    const layer2 = document.createElement("div");
    layer2.id = "layer2"
    const layer2Features = model.layers[2].weights[0].read().arraySync();
    const layer2bias = model.layers[2].weights[1].read().arraySync();
    console.log("layer2weights",layer2Features);
    console.log("layer2bias",layer2bias);
    for(var j=0;j<8;j++){
      for(var i =0;i<16;i++){
        const newBox = document.createElement("canvas");
        
        newBox.width = 5;
        newBox.height = 5;
        const newBoxCtx = newBox.getContext("2d");
        var imageData = newBoxCtx.createImageData(5, 5);

        newBoxCtx.fillStyle = "green";
        newBoxCtx.fillRect(0, 0, 5, 5);
        var leastVal = 0
        var greatestVal = 0
        for(var x = 0;x<5;x++){
          for(var y=0;y<5;y++){
            var weight = layer2Features[x][y][j][i];
            var bias = layer2bias[i];
            greatestVal = Math.max(
              Math.abs( 255*(weight+bias)), greatestVal
            )

          }
        }
        for(var x = 0;x<5;x++){
          for(var y=0;y<5;y++){
            var weight = layer2Features[x][y][j][i];
            var bias = layer2bias[i];
            v = 255*(weight+bias) * (255/greatestVal)
            var r = v;
            if( v < 1){
              r = v*-1;
            }
            imageData.data[(x*5 + y)*4+0] = r
            imageData.data[(x*5 + y)*4+1] = v
            imageData.data[(x*5 + y)*4+2] = v
            imageData.data[(x*5 + y)*4+3] = 255
          }
        }
        newBoxCtx.putImageData(imageData, 0, 0);
        newBox.className = "featureCanvas";
        layer2.appendChild(newBox);
      }
      layer2.appendChild(document.createElement("div"));
    }

    document.body.appendChild(layer2);

    const layer2output = document.createElement("div");
    layer2output.id = "layer2output"
    for(var i =0;i<16;i++){
      const newBox = document.createElement("canvas");
      
      newBox.width = 8;
      newBox.height = 8;
      const newBoxCtx = newBox.getContext("2d");

      newBoxCtx.fillStyle = "black";
      newBoxCtx.fillRect(0, 0, 8, 8);


      newBox.className = "layer2outputCanvas";
      layer2output.appendChild(newBox);
    }
    document.body.appendChild(layer2output);
    document.body.appendChild(document.createElement("br"));
    const layer3output = document.createElement("div");
    layer3output.id = "layer3output"
    for(var i =0;i<16;i++){
      const newBox = document.createElement("canvas");
      
      newBox.width = 4;
      newBox.height = 4;
      const newBoxCtx = newBox.getContext("2d");

      newBoxCtx.fillStyle = "black";
      newBoxCtx.fillRect(0, 0, 4, 4);


      newBox.className = "layer3outputCanvas";
      layer3output.appendChild(newBox);
    }
    document.body.appendChild(layer3output);
    updateCanvas()
}
async function run() {
  const trainModel = false;
  //const data = new TrainingData();
  //await data.load();

  // await showExamples(data);
  if(trainModel){
      model = getModel();
      tfvis.show.modelSummary({name: 'Model Architecture', tab: 'Model'}, model);
      await train(model, data);
      await showAccuracy(model, data);
      await showConfusion(model, data);
      await model.save('localstorage://my-model');
      //await model.save('downloads://model');
    }else{
      //model = await tf.loadLayersModel('localstorage://my-model');
      model = await tf.loadLayersModel('./model.json');
    }

    //tfvis.show.modelSummary({name: 'Model Architecture', tab: 'Model'}, model);
    //doPrediction(model, data, 1)
    await startUp();



}




document.addEventListener('DOMContentLoaded', run);


const classNames = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine'];

function doPrediction(model, data, testDataSize = 500) {
  const IMAGE_WIDTH = 28;
  const IMAGE_HEIGHT = 28;
  const testData = data.nextTestBatch(testDataSize);
  const testxs = testData.xs.reshape([testDataSize, IMAGE_WIDTH, IMAGE_HEIGHT, 1]);
  const labels = testData.labels.argMax(-1);
  const preds = model.predict(testxs).argMax(-1);

  testxs.dispose();
  return [preds, labels];
}


async function showAccuracy(model, data) {
  const [preds, labels] = doPrediction(model, data);
  const classAccuracy = await tfvis.metrics.perClassAccuracy(labels, preds);
  const container = {name: 'Accuracy', tab: 'Evaluation'};
  tfvis.show.perClassAccuracy(container, classAccuracy, classNames);

  labels.dispose();
}

async function showConfusion(model, data) {
  const [preds, labels] = doPrediction(model, data);
  const confusionMatrix = await tfvis.metrics.confusionMatrix(labels, preds);
  const container = {name: 'Confusion Matrix', tab: 'Evaluation'};
  tfvis.render.confusionMatrix(container, {values: confusionMatrix, tickLabels: classNames});

  labels.dispose();
}


