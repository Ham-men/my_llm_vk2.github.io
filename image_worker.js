importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.21.0/dist/tf.min.js');

let models = [];
let modelsLoaded = false;

async function ensureModelsLoaded(taskId) {
  if (modelsLoaded) { console.log(`[Worker ${taskId}] Models already loaded`); return; }
  console.log(`[Worker ${taskId}] Loading models...`);
  const modelUrls = [
    './web_model_1/model.json',
    './web_model_2/model.json',
    './web_model_3/model.json',
  ];
  for (let i = 0; i < modelUrls.length; i++) {
    postMessage({ type: 'progress', taskId, status: 'loading_model', progress: Math.round(((i) / modelUrls.length) * 100) });
    try {
      const m = await tf.loadGraphModel(modelUrls[i]);
      models.push(m);
      console.log(`[Worker ${taskId}] Model ${i+1} loaded`);
    } catch (e) {
      console.error(`[Worker ${taskId}] Failed to load model ${i+1}:`, e);
      throw new Error(`Model ${i+1} load failed: ${e.message}`);
    }
  }
  modelsLoaded = true;
  console.log(`[Worker ${taskId}] All models loaded`);
}

function createThumbnail(imageBitmap, size) {
  const canvas = new OffscreenCanvas(size, size);
  const ctx = canvas.getContext('2d');
  ctx.drawImage(imageBitmap, 0, 0, size, size);
  return ctx.getImageData(0, 0, size, size);
}

function imageDataToTensor(imageData) {
  const data = new Float32Array(224 * 224 * 3);
  const src = imageData.data;
  for (let i = 0; i < 224 * 224; i++) {
    data[i * 3] = src[i * 4] / 255;
    data[i * 3 + 1] = src[i * 4 + 1] / 255;
    data[i * 3 + 2] = src[i * 4 + 2] / 255;
  }
  return tf.tensor4d(data, [1, 224, 224, 3]);
}

async function predictEnsemble(thumbnailTensor) {
  console.log(`[Worker] Predicting ensemble...`);
  const sums = [0, 0, 0];
  for (let i = 0; i < models.length; i++) {
    try {
      const out = await models[i].predict(thumbnailTensor);
      const vals = await Promise.all(out.map(t => t.data()));
      tf.dispose(out);
      sums[0] += vals[0][0];
      sums[1] += vals[1][0];
      sums[2] += vals[2][0];
      console.log(`[Worker] Model ${i+1}: brightness=${vals[0][0].toFixed(4)}, contrast=${vals[1][0].toFixed(4)}, saturation=${vals[2][0].toFixed(4)}`);
    } catch (e) {
      console.error(`[Worker] Model ${i+1} predict error:`, e);
      throw e;
    }
  }
  const avg = [sums[0] / models.length, sums[1] / models.length, sums[2] / models.length];
  console.log(`[Worker] Ensemble avg: brightness=${avg[0].toFixed(4)}, contrast=${avg[1].toFixed(4)}, saturation=${avg[2].toFixed(4)}`);
  return avg;
}

function applyAdjustments(imageBitmap, params, taskId, signal) {
  const [brightness, contrast, saturation] = params;
  const w = imageBitmap.width, h = imageBitmap.height;
  console.log(`[Worker ${taskId}] Applying adjustments: ${w}x${h}, params=[${params.map(v=>v.toFixed(4))}]`);
  const canvas = new OffscreenCanvas(w, h);
  const ctx = canvas.getContext('2d');
  ctx.drawImage(imageBitmap, 0, 0);
  const imageData = ctx.getImageData(0, 0, w, h);
  const data = imageData.data;

  const bAdj = brightness * 50;
  const cFactor = contrast;
  const sAdj = saturation;

  const totalPixels = w * h;
  const chunkSize = Math.max(1000, Math.floor(totalPixels / 30));
  const cMul = (259 * (cFactor * 128 + 255)) / (255 * (259 - cFactor * 128));

  for (let i = 0; i < data.length; i += 4) {
    let r = data[i], g = data[i + 1], b = data[i + 2];

    if (sAdj !== 0) {
      const max = Math.max(r, g, b) / 255;
      const min = Math.min(r, g, b) / 255;
      const delta = max - min;
      if (delta > 0.001) {
        const L = (max + min) / 2;
        const S = L > 0.5 ? delta / (2 - max - min) : delta / (max + min);
        const newS = Math.min(1, Math.max(0, S * (1 + sAdj)));
        if (newS > 0 && S > 0) {
          const ratio = newS / S;
          const rF = r / 255, gF = g / 255, bF = b / 255;
          const newR = L > 0.5
            ? L + (rF - L) * (2 - max - min) * ratio / delta
            : L + (rF - L) * (max + min) * ratio / delta;
          const newG = L > 0.5
            ? L + (gF - L) * (2 - max - min) * ratio / delta
            : L + (gF - L) * (max + min) * ratio / delta;
          const newB = L > 0.5
            ? L + (bF - L) * (2 - max - min) * ratio / delta
            : L + (bF - L) * (max + min) * ratio / delta;
          r = Math.round(Math.max(0, Math.min(255, newR * 255)));
          g = Math.round(Math.max(0, Math.min(255, newG * 255)));
          b = Math.round(Math.max(0, Math.min(255, newB * 255)));
        }
      }
    }

    if (cFactor !== 0) {
      r = Math.max(0, Math.min(255, cMul * (r - 128) + 128));
      g = Math.max(0, Math.min(255, cMul * (g - 128) + 128));
      b = Math.max(0, Math.min(255, cMul * (b - 128) + 128));
    }

    if (bAdj !== 0) {
      r = Math.max(0, Math.min(255, r + bAdj));
      g = Math.max(0, Math.min(255, g + bAdj));
      b = Math.max(0, Math.min(255, b + bAdj));
    }

    data[i] = r;
    data[i + 1] = g;
    data[i + 2] = b;

    const processed = (i / 4) + 1;
    if (processed % chunkSize === 0) {
      const pct = Math.round(50 + (processed / totalPixels) * 45);
      postMessage({ type: 'progress', taskId, status: 'applying', progress: pct });
      if (signal && signal.aborted) throw new DOMException('Aborted', 'AbortError');
    }
  }

  ctx.putImageData(imageData, 0, 0);
  const result = canvas.transferToImageBitmap();
  console.log(`[Worker ${taskId}] Adjustments applied`);
  return result;
}

self.onmessage = async (e) => {
  const { type, taskId, imageBlob } = e.data;
  console.log(`[Worker] Received message: type=${type}, taskId=${taskId}, blob=${imageBlob ? imageBlob.size + 'bytes' : 'null'}`);

  if (type === 'process') {
    const controller = new AbortController();
    self.__controller = controller;

    try {
      postMessage({ type: 'progress', taskId, status: 'loading_model', progress: 0 });
      await ensureModelsLoaded(taskId);
      postMessage({ type: 'progress', taskId, status: 'decoding', progress: 20 });

      console.log(`[Worker ${taskId}] Decoding image...`);
      const bitmap = await createImageBitmap(imageBlob);
      console.log(`[Worker ${taskId}] Decoded: ${bitmap.width}x${bitmap.height}`);
      postMessage({ type: 'progress', taskId, status: 'thumbnail', progress: 30 });

      const thumb = createThumbnail(bitmap, 224);
      const tensor = imageDataToTensor(thumb);
      postMessage({ type: 'progress', taskId, status: 'predicting', progress: 40 });

      const params = await predictEnsemble(tensor);
      tf.dispose(tensor);
      postMessage({ type: 'progress', taskId, status: 'applying', progress: 50 });

      const resultBitmap = applyAdjustments(bitmap, params, taskId, controller.signal);

      postMessage({ type: 'progress', taskId, status: 'encoding', progress: 95 });
      console.log(`[Worker ${taskId}] Encoding result...`);
      const resultCanvas = new OffscreenCanvas(resultBitmap.width, resultBitmap.height);
      resultCanvas.getContext('2d').drawImage(resultBitmap, 0, 0);
      const blob = await resultCanvas.convertToBlob({ type: 'image/jpeg', quality: 0.95 });
      console.log(`[Worker ${taskId}] Result encoded: ${blob.size} bytes`);

      postMessage({ type: 'result', taskId, blob, params });
      console.log(`[Worker ${taskId}] Done`);
    } catch (err) {
      console.error(`[Worker ${taskId}] Error:`, err.name, err.message, err.stack);
      if (err.name === 'AbortError') {
        postMessage({ type: 'canceled', taskId });
      } else {
        postMessage({ type: 'error', taskId, error: err.message + ' | ' + err.stack });
      }
    }
  } else if (type === 'cancel') {
    console.log(`[Worker] Cancel requested for ${taskId}`);
    if (self.__controller) {
      self.__controller.abort();
    }
  }
};
