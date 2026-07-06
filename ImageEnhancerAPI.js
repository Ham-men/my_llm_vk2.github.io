class ImageEnhancerAPI {
  constructor(maxConcurrent = 2) {
    this.tasks = new Map();
    this.nextId = 1;
    this.listeners = {};
    this.maxConcurrent = maxConcurrent;
    this.activeCount = 0;
    this.queue = [];
    this.worker = null;
  }

  addTask(imageBlob, fileName) {
    const id = `task_${this.nextId++}`;
    const task = {
      id,
      fileName: fileName || `image_${id}.jpg`,
      status: 'queued',
      progress: 0,
      imageBlob,
      resultBlob: null,
      params: null,
      createdAt: Date.now(),
      completedAt: null,
      error: null,
    };
    this.tasks.set(id, task);
    this._emit('taskCreated', task);
    this._processNext();
    return id;
  }

  addTasks(imageBlobs, fileNames) {
    const ids = [];
    for (let i = 0; i < imageBlobs.length; i++) {
      const name = fileNames && fileNames[i] ? fileNames[i] : `image_${i}.jpg`;
      ids.push(this.addTask(imageBlobs[i], name));
    }
    return ids;
  }

  getStatus(taskId) {
    const task = this.tasks.get(taskId);
    if (!task) return null;
    return {
      id: task.id,
      status: task.status,
      progress: task.progress,
      fileName: task.fileName,
    };
  }

  async getResult(taskId) {
    const task = this.tasks.get(taskId);
    if (!task || task.status !== 'completed') return null;
    return task.resultBlob;
  }

  getResultParams(taskId) {
    const task = this.tasks.get(taskId);
    if (!task) return null;
    return { params: task.params, fileName: task.fileName };
  }

  cancelTask(taskId) {
    const task = this.tasks.get(taskId);
    if (!task) return false;
    if (task.status === 'processing' && this.activeTaskId === taskId) {
      if (this.worker) {
        this.worker.postMessage({ type: 'cancel', taskId });
      }
    }
    task.status = 'canceled';
    this._emit('taskUpdated', task);
    return true;
  }

  getAllTasks() {
    return Array.from(this.tasks.values()).map(t => ({
      id: t.id, status: t.status, progress: t.progress, fileName: t.fileName,
    }));
  }

  on(eventName, callback) {
    if (!this.listeners[eventName]) this.listeners[eventName] = [];
    this.listeners[eventName].push(callback);
  }

  off(eventName, callback) {
    const cbs = this.listeners[eventName];
    if (cbs) {
      this.listeners[eventName] = cbs.filter(cb => cb !== callback);
    }
  }

  _emit(eventName, data) {
    (this.listeners[eventName] || []).forEach(cb => cb(data));
  }

  _processNext() {
    if (this.activeCount >= this.maxConcurrent) return;
    const next = Array.from(this.tasks.values()).find(t => t.status === 'queued');
    if (!next) return;
    this.activeCount++;
    next.status = 'processing';
    next.progress = 0;
    this._emit('taskUpdated', next);
    this._startWorker(next);
  }

  _startWorker(task) {
    this.activeTaskId = task.id;
    console.log(`[API] Starting worker for task ${task.id}, file=${task.fileName}`);
    this.worker = new Worker('image_worker.js');

    this.worker.onmessage = (e) => {
      const msg = e.data;
      console.log(`[API] Worker message: type=${msg.type}, taskId=${msg.taskId}`, msg.type === 'error' ? `error=${msg.error}` : '');
      const t = this.tasks.get(msg.taskId);
      if (!t) {
        console.warn(`[API] Task ${msg.taskId} not found for message type=${msg.type}`);
        return;
      }

      if (msg.type === 'progress') {
        t.status = msg.status;
        t.progress = msg.progress;
        this._emit('taskUpdated', t);
      } else if (msg.type === 'result') {
        t.status = 'completed';
        t.progress = 100;
        t.resultBlob = msg.blob;
        t.params = msg.params || null;
        t.completedAt = Date.now();
        console.log(`[API] Task ${task.id} completed, blob=${msg.blob ? msg.blob.size + 'bytes' : 'null'}, params=${JSON.stringify(msg.params)}`);
        this._emit('taskUpdated', t);
        this._cleanupWorker();
      } else if (msg.type === 'canceled') {
        console.log(`[API] Task ${task.id} canceled`);
        t.status = 'canceled';
        this._emit('taskUpdated', t);
        this._cleanupWorker();
      } else if (msg.type === 'error') {
        console.error(`[API] Task ${task.id} error: ${msg.error}`);
        t.status = 'error';
        t.error = msg.error;
        this._emit('taskUpdated', t);
        this._cleanupWorker();
      }
    };

    this.worker.onerror = (err) => {
      console.error(`[API] Worker unhandled error for task ${task.id}:`, err.message, err.lineno, err.filename);
      const t = this.tasks.get(task.id);
      if (t) {
        t.status = 'error';
        t.error = `Worker error: ${err.message} (line ${err.lineno})`;
        this._emit('taskUpdated', t);
      }
      this._cleanupWorker();
    };

    console.log(`[API] Sending process message to worker for task ${task.id}`);
    this.worker.postMessage({ type: 'process', taskId: task.id, imageBlob: task.imageBlob });
  }

  _cleanupWorker() {
    if (this.worker) {
      this.worker.terminate();
      this.worker = null;
    }
    this.activeTaskId = null;
    this.activeCount--;
    this._processNext();
  }
}
