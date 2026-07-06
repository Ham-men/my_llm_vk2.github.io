importScripts(
  'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js',
  'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@4.22.0/dist/tf-backend-wasm.min.js'
);

let models = [];
let modelsLoaded = false;

async function ensureTFJSReady(taskId) {
  console.log(`[Worker ${taskId}] TFJS version: ${tf.version.tfjs}`);
  await tf.ready();
  let backend = tf.getBackend();
  console.log(`[Worker ${taskId}] Default backend: ${backend}`);
  console.log(`[Worker ${taskId}] Available backends:`, tf.engine().backendNames());

  if (backend === 'webgl') {
    console.log(`[Worker ${taskId}] WebGL available, using it for inference`);
    return;
  }

  if (backend === 'cpu') {
    tf.wasm().setWasmPath('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@4.22.0/wasm-out/');
    try {
      await tf.setBackend('wasm');
      console.log(`[Worker ${taskId}] Switched to WASM backend`);
    } catch (e) {
      console.warn(`[Worker ${taskId}] WASM unavailable: ${e.message}`);
    }
  }
}

async function ensureModelsLoaded(taskId) {
  if (modelsLoaded) { return; }
  console.log(`[Worker ${taskId}] Loading models...`);

  await ensureTFJSReady(taskId);

  const modelUrls = [
    './web_model_1/model.json',
    './web_model_2/model.json',
    './web_model_3/model.json',
  ];
  for (let i = 0; i < modelUrls.length; i++) {
    postMessage({ type: 'progress', taskId, status: 'loading_model', progress: Math.round(((i) / modelUrls.length) * 100) });
    try {
      const m = await tf.loadGraphModel(modelUrls[i], { fromTFHub: false });
      console.log(`[Worker ${taskId}] Model ${i+1} loaded`);
      console.log(`[Worker ${taskId}] Model ${i+1} input:`, JSON.stringify(m.inputs));
      console.log(`[Worker ${taskId}] Model ${i+1} output:`, JSON.stringify(m.outputs));
      models.push(m);
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
  console.log(`[Worker] Predict ensemble input shape: ${thumbnailTensor.shape}`);
  const sums = [0, 0, 0];

  for (let i = 0; i < models.length; i++) {
    console.log(`[Worker] Model ${i+1} executing...`);

    const out = await models[i].executeAsync(thumbnailTensor);
    console.log(`[Worker] Model ${i+1} output type: ${typeof out}`);

    let tensors = [];
    if (out instanceof tf.Tensor) {
      tensors = [out];
    } else if (out && typeof out === 'object') {
      if (out.constructor && out.constructor.name === 'Map') {
        tensors = Array.from(out.values());
      } else {
        tensors = Object.values(out);
      }
    }
    console.log(`[Worker] Model ${i+1} got ${tensors.length} output tensors`);

    for (let j = 0; j < Math.min(3, tensors.length); j++) {
      const vals = await tensors[j].data();
      sums[j] += vals[0] || 0;
      console.log(`[Worker] Model ${i+1} out[${j}] = ${(vals[0]||0).toFixed(4)}`);
      tensors[j].dispose();
    }
  }

  const avg = sums.map(s => s / models.length);
  console.log(`[Worker] Ensemble: brightness=${avg[0].toFixed(4)} contrast=${avg[1].toFixed(4)} saturation=${avg[2].toFixed(4)}`);
  return avg;
}

function applyAdjustments(imageBitmap, params, taskId, signal) {
  const [brightness, contrast, saturation] = params;
  const w = imageBitmap.width, h = imageBitmap.height;
  const canvas = new OffscreenCanvas(w, h);
  const ctx = canvas.getContext('2d');
  ctx.drawImage(imageBitmap, 0, 0);
  const imageData = ctx.getImageData(0, 0, w, h);
  const data = imageData.data;

  const bAdj = brightness * 60;
  const cFactor = contrast;
  const sAdj = saturation;

  const totalPixels = w * h;
  const chunkPixels = Math.max(1000, Math.floor(totalPixels / 30));
  const cMul = (259 * (cFactor * 128 + 255)) / (255 * (259 - cFactor * 128));

  for (let i = 0; i < data.length; i += 4) {
    let r = data[i], g = data[i + 1], b = data[i + 2];

    if (Math.abs(sAdj) > 0.001) {
      const rf = r / 255, gf = g / 255, bf = b / 255;
      const max = Math.max(rf, gf, bf), min = Math.min(rf, gf, bf);
      const delta = max - min;
      if (delta > 0.001) {
        const L = (max + min) / 2;
        const S = L > 0.5 ? delta / (2 - max - min) : delta / (max + min);
        const newS = Math.min(1, Math.max(0.001, S * (1 + sAdj)));
        const ratio = newS / S;
        const newR = L + (rf - L) * ratio;
        const newG = L + (gf - L) * ratio;
        const newB = L + (bf - L) * ratio;
        r = Math.round(Math.max(0, Math.min(255, newR * 255)));
        g = Math.round(Math.max(0, Math.min(255, newG * 255)));
        b = Math.round(Math.max(0, Math.min(255, newB * 255)));
      }
    }

    if (Math.abs(cFactor) > 0.001) {
      r = Math.max(0, Math.min(255, (cMul * (r - 128) + 128)));
      g = Math.max(0, Math.min(255, (cMul * (g - 128) + 128)));
      b = Math.max(0, Math.min(255, (cMul * (b - 128) + 128)));
    }

    if (Math.abs(bAdj) > 0.5) {
      r = Math.max(0, Math.min(255, r + bAdj));
      g = Math.max(0, Math.min(255, g + bAdj));
      b = Math.max(0, Math.min(255, b + bAdj));
    }

    data[i] = r; data[i + 1] = g; data[i + 2] = b;

    const px = (i / 4) + 1;
    if (px % chunkPixels === 0) {
      const pct = Math.round(50 + (px / totalPixels) * 45);
      postMessage({ type: 'progress', taskId, status: 'applying', progress: pct });
      if (signal?.aborted) throw new DOMException('Aborted', 'AbortError');
    }
  }

  ctx.putImageData(imageData, 0, 0);
  return canvas.transferToImageBitmap();
}

self.onmessage = async (e) => {
  const { type, taskId, imageBlob } = e.data;

  if (type === 'process') {
    const controller = new AbortController();
    self.__controller = controller;

    try {
      postMessage({ type: 'progress', taskId, status: 'loading_model', progress: 0 });
      await ensureModelsLoaded(taskId);

      postMessage({ type: 'progress', taskId, status: 'decoding', progress: 20 });
      const bitmap = await createImageBitmap(imageBlob);
      postMessage({ type: 'progress', taskId, status: 'thumbnail', progress: 30 });

      const thumb = createThumbnail(bitmap, 224);
      const tensor = imageDataToTensor(thumb);
      postMessage({ type: 'progress', taskId, status: 'predicting', progress: 40 });

      const params = await predictEnsemble(tensor);
      tf.dispose(tensor);
      postMessage({ type: 'progress', taskId, status: 'applying', progress: 50 });

      const resultBitmap = applyAdjustments(bitmap, params, taskId, controller.signal);
      postMessage({ type: 'progress', taskId, status: 'encoding', progress: 95 });

      const resultCanvas = new OffscreenCanvas(resultBitmap.width, resultBitmap.height);
      resultCanvas.getContext('2d').drawImage(resultBitmap, 0, 0);
      const blob = await resultCanvas.convertToBlob({ type: 'image/jpeg', quality: 0.95 });

      postMessage({ type: 'result', taskId, blob, params });
    } catch (err) {
      console.error(`[Worker ${taskId}] Error: ${err.name}: ${err.message}`, err.stack);
      if (err.name === 'AbortError') {
        postMessage({ type: 'canceled', taskId });
      } else {
        postMessage({ type: 'error', taskId, error: `${err.name}: ${err.message}\n${err.stack}` });
      }
    }
  } else if (type === 'cancel') {
    if (self.__controller) { self.__controller.abort(); }
  }
};
