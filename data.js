const IMAGE_SIZE = 784;
const NUM_CLASSES = 10;
const NUM_DATASET_ELEMENTS = 65000;

const NUM_TRAIN_ELEMENTS = 55000;
const NUM_TEST_ELEMENTS = NUM_DATASET_ELEMENTS - NUM_TRAIN_ELEMENTS;

const MNIST_DATA_PATH =
    './mnist_images_uint8_bw';
const MNIST_LABELS_PATH =
    './mnist_labels_uint8';


export class TrainingData {
    constructor() {
      this.shuffledTrainIndex = 0;
      this.shuffledTestIndex = 0;
    }
  
    async load() {
  
      const imgRequest = fetch(MNIST_DATA_PATH)
      const labelsRequest = fetch(MNIST_LABELS_PATH);
      const [imgResponse, labelsResponse] =
          await Promise.all([imgRequest, labelsRequest]);
  

      this.datasetImages = new Uint8Array(await imgResponse.arrayBuffer());
      this.datasetLabels = new Uint8Array(await labelsResponse.arrayBuffer());
      this.trainIndices = tf.util.createShuffledIndices(NUM_TRAIN_ELEMENTS);
      this.testIndices = tf.util.createShuffledIndices(NUM_TEST_ELEMENTS);
  
      this.trainImages =
          this.datasetImages.slice(0, IMAGE_SIZE * NUM_TRAIN_ELEMENTS);
      this.testImages = this.datasetImages.slice(IMAGE_SIZE * NUM_TRAIN_ELEMENTS);
      this.trainLabels =
          this.datasetLabels.slice(0, NUM_CLASSES * NUM_TRAIN_ELEMENTS);
      this.testLabels =
          this.datasetLabels.slice(NUM_CLASSES * NUM_TRAIN_ELEMENTS);
    }
  
    nextTrainBatch(batchSize) {
      return this.nextBatch(
          batchSize, [this.trainImages, this.trainLabels], () => {
            this.shuffledTrainIndex =
                (this.shuffledTrainIndex + 1) % this.trainIndices.length;
            return this.trainIndices[this.shuffledTrainIndex];
          });
    }
  
    nextTestBatch(batchSize) {
      return this.nextBatch(batchSize, [this.testImages, this.testLabels], () => {
        this.shuffledTestIndex =
            (this.shuffledTestIndex + 1) % this.testIndices.length;
        return this.testIndices[this.shuffledTestIndex];
      });
    }
  
    nextBatch(batchSize, data, index) {
      const batchDataArray = new Uint8Array(batchSize * IMAGE_SIZE);
      const batchLabelsArray = new Uint8Array(batchSize * NUM_CLASSES);
  
      for (let i = 0; i < batchSize; i++) {
        const idx = index();
  
        const image =
            data[0].slice(idx * IMAGE_SIZE, idx * IMAGE_SIZE + IMAGE_SIZE);
        batchDataArray.set(image, i * IMAGE_SIZE);
  
        const label =
            data[1].slice(idx * NUM_CLASSES, idx * NUM_CLASSES + NUM_CLASSES);
        batchLabelsArray.set(label, i * NUM_CLASSES);
      }
  
      const xs = tf.tensor2d(batchDataArray, [batchSize, IMAGE_SIZE]);
      const labels = tf.tensor2d(batchLabelsArray, [batchSize, NUM_CLASSES]);
  
      return {xs, labels};
    }
  }
  