const MNIST_IMAGES_SPRITE_PATH = 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png';
const MNIST_LABELS_PATH = 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8';

const IMAGE_W = 28;
const IMAGE_H = 28;
const IMAGE_SIZE = IMAGE_W * IMAGE_H;
const NUM_CLASSES = 10;

const CHUNK_SIZE = 5000;
const TOTAL_ELEMENTS = 10000;
const NUM_TRAIN = 8000;
const NUM_TEST = TOTAL_ELEMENTS - NUM_TRAIN;

const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const progressFill = document.getElementById('progressFill');
const guessBtn = document.getElementById('guessBtn');
const eraseBtn = document.getElementById('eraseBtn');
const bigDigit = document.getElementById('bigDigit');
const confidenceNote = document.getElementById('confidenceNote');
const barsEl = document.getElementById('bars');
const canvas = document.getElementById('draw');
const ctx = canvas.getContext('2d');

function setStatus(text, mode) {
    statusText.textContent = text;
    statusDot.className = 'status-dot' + (mode ? ' ' + mode : '');
}

function setProgress(pct) {
    progressFill.style.width =
        Math.max(2, Math.min(100, pct)) + '%';
}

// Build the 10 confidence bar rows
for (let d = 0; d < 10; d++) {
    const row = document.createElement('div');

    row.className = 'bar-row';

    row.innerHTML = `
        <span>${d}</span>
        <div class="bar-track">
            <div class="bar-fill" id="fill-${d}"></div>
        </div>
        <span class="bar-pct" id="pct-${d}">0%</span>
    `;

    barsEl.appendChild(row);
}


// --------------------------------------------------
// Load MNIST dataset
// --------------------------------------------------

async function loadMnistSubset() {

    const img = new Image();

    img.crossOrigin = 'anonymous';

    const imgReady = new Promise((res, rej) => {
        img.onload = res;
        img.onerror = rej;
    });

    img.src = MNIST_IMAGES_SPRITE_PATH;

    await imgReady;


    const spriteCanvas = document.createElement('canvas');

    spriteCanvas.width = img.width;
    spriteCanvas.height = CHUNK_SIZE;

    const sctx = spriteCanvas.getContext('2d');


    const pixels =
        new Float32Array(TOTAL_ELEMENTS * IMAGE_SIZE);

    const chunksNeeded =
        Math.ceil(TOTAL_ELEMENTS / CHUNK_SIZE);


    for (let c = 0; c < chunksNeeded; c++) {

        sctx.clearRect(
            0,
            0,
            spriteCanvas.width,
            CHUNK_SIZE
        );

        sctx.drawImage(
            img,
            0,
            c * CHUNK_SIZE,
            img.width,
            CHUNK_SIZE,
            0,
            0,
            img.width,
            CHUNK_SIZE
        );


        const data = sctx.getImageData(
            0,
            0,
            spriteCanvas.width,
            CHUNK_SIZE
        ).data;


        const rowsInThisChunk =
            Math.min(
                CHUNK_SIZE,
                TOTAL_ELEMENTS - c * CHUNK_SIZE
            );


        const offset =
            c * CHUNK_SIZE * IMAGE_SIZE;


        const valuesInChunk =
            rowsInThisChunk * IMAGE_SIZE;


        for (let j = 0; j < valuesInChunk; j++) {

            pixels[offset + j] =
                data[j * 4] / 255;
        }
    }


    const labelsResp =
        await fetch(MNIST_LABELS_PATH);

    const labelsBuf =
        await labelsResp.arrayBuffer();

    const allLabels =
        new Uint8Array(labelsBuf);


    const labels =
        allLabels.slice(
            0,
            TOTAL_ELEMENTS * NUM_CLASSES
        );


    // Shuffle the data
    const order =
        Array.from(
            {
                length: TOTAL_ELEMENTS
            },
            (_, i) => i
        );


    for (
        let i = order.length - 1;
        i > 0;
        i--
    ) {

        const j =
            Math.floor(
                Math.random() * (i + 1)
            );

        [
            order[i],
            order[j]
        ] = [
            order[j],
            order[i]
        ];
    }


    const shuffledPixels =
        new Float32Array(
            TOTAL_ELEMENTS * IMAGE_SIZE
        );


    const shuffledLabels =
        new Uint8Array(
            TOTAL_ELEMENTS * NUM_CLASSES
        );


    order.forEach((srcIdx, dstIdx) => {

        shuffledPixels.set(
            pixels.subarray(
                srcIdx * IMAGE_SIZE,
                (srcIdx + 1) * IMAGE_SIZE
            ),
            dstIdx * IMAGE_SIZE
        );


        shuffledLabels.set(
            labels.subarray(
                srcIdx * NUM_CLASSES,
                (srcIdx + 1) * NUM_CLASSES
            ),
            dstIdx * NUM_CLASSES
        );
    });


    const xTrain =
        tf.tensor4d(
            shuffledPixels.subarray(
                0,
                NUM_TRAIN * IMAGE_SIZE
            ),
            [
                NUM_TRAIN,
                IMAGE_W,
                IMAGE_H,
                1
            ]
        );


    const yTrain =
        tf.tensor2d(
            shuffledLabels.subarray(
                0,
                NUM_TRAIN * NUM_CLASSES
            ),
            [
                NUM_TRAIN,
                NUM_CLASSES
            ]
        );


    const xTest =
        tf.tensor4d(
            shuffledPixels.subarray(
                NUM_TRAIN * IMAGE_SIZE
            ),
            [
                NUM_TEST,
                IMAGE_W,
                IMAGE_H,
                1
            ]
        );


    const yTest =
        tf.tensor2d(
            shuffledLabels.subarray(
                NUM_TRAIN * NUM_CLASSES
            ),
            [
                NUM_TEST,
                NUM_CLASSES
            ]
        );


    return {
        xTrain,
        yTrain,
        xTest,
        yTest
    };
}


// --------------------------------------------------
// Build CNN model
// --------------------------------------------------

function buildModel() {

    const model = tf.sequential();


    model.add(
        tf.layers.conv2d({
            inputShape: [28, 28, 1],
            filters: 8,
            kernelSize: 5,
            activation: 'relu',
            padding: 'same'
        })
    );


    model.add(
        tf.layers.maxPooling2d({
            poolSize: 2
        })
    );


    model.add(
        tf.layers.conv2d({
            filters: 16,
            kernelSize: 5,
            activation: 'relu',
            padding: 'same'
        })
    );


    model.add(
        tf.layers.maxPooling2d({
            poolSize: 2
        })
    );


    model.add(
        tf.layers.flatten()
    );


    model.add(
        tf.layers.dense({
            units: 64,
            activation: 'relu'
        })
    );


    model.add(
        tf.layers.dropout({
            rate: 0.25
        })
    );


    model.add(
        tf.layers.dense({
            units: 10,
            activation: 'softmax'
        })
    );


    model.compile({
        optimizer: 'adam',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });


    return model;
}


let trainedModel = null;


// --------------------------------------------------
// Main training function
// --------------------------------------------------

async function main() {

    try {

        setStatus(
            'Fetching 10,000 handwritten digits…'
        );

        setProgress(8);


        const {
            xTrain,
            yTrain,
            xTest,
            yTest
        } = await loadMnistSubset();


        setStatus(
            'Building the network…'
        );

        setProgress(18);


        const model = buildModel();


        const epochs = 6;

        const batchesPerEpoch =
            Math.ceil(
                (NUM_TRAIN * 0.9) / 128
            );


        let batchesSeen = 0;

        const totalBatches =
            epochs * batchesPerEpoch;


        setStatus(
            `Training — epoch 1 of ${epochs}…`
        );


        await model.fit(
            xTrain,
            yTrain,
            {

                epochs: epochs,

                batchSize: 128,

                validationSplit: 0.1,

                callbacks: {

                    onEpochBegin: (epoch) => {

                        setStatus(
                            `Training — epoch ${epoch + 1} of ${epochs}…`
                        );
                    },


                    onBatchEnd: () => {

                        batchesSeen++;


                        setProgress(
                            18 +
                            (batchesSeen / totalBatches) *
                            70
                        );
                    }
                }
            }
        );


        setStatus(
            'Checking its work on digits it hasn’t seen…'
        );


        const evalResult =
            model.evaluate(
                xTest,
                yTest
            );


        const testAcc =
            (await evalResult[1].data())[0];


        setProgress(100);


        setStatus(
            `Ready — about ${(testAcc * 100).toFixed(1)}% accurate on digits it never trained on.`,
            'done'
        );


        trainedModel = model;


        guessBtn.disabled = false;
        eraseBtn.disabled = false;


        bigDigit.textContent =
            'draw something';


        tf.dispose([
            xTrain,
            yTrain,
            xTest,
            yTest
        ]);

    } catch (err) {

        console.error(err);


        setStatus(
            'Could not load the training data — check your connection and reload.',
            'error'
        );
    }
}


// --------------------------------------------------
// Drawing setup
// --------------------------------------------------

ctx.fillStyle = '#000';

ctx.fillRect(
    0,
    0,
    canvas.width,
    canvas.height
);


ctx.lineWidth = 18;

ctx.lineCap = 'round';

ctx.lineJoin = 'round';

ctx.strokeStyle = '#fff';


let drawing = false;

let lastX = 0;

let lastY = 0;


function pointerPos(e) {

    const rect =
        canvas.getBoundingClientRect();


    const scaleX =
        canvas.width / rect.width;


    const scaleY =
        canvas.height / rect.height;


    const clientX =
        e.touches
            ? e.touches[0].clientX
            : e.clientX;


    const clientY =
        e.touches
            ? e.touches[0].clientY
            : e.clientY;


    return {
        x: (clientX - rect.left) * scaleX,
        y: (clientY - rect.top) * scaleY
    };
}


// Start drawing
function startDraw(e) {

    e.preventDefault();

    drawing = true;


    const p =
        pointerPos(e);


    lastX = p.x;

    lastY = p.y;


    ctx.beginPath();


    ctx.arc(
        p.x,
        p.y,
        ctx.lineWidth / 2,
        0,
        Math.PI * 2
    );


    ctx.fillStyle = '#fff';

    ctx.fill();
}


// Draw
function moveDraw(e) {

    if (!drawing) return;

    e.preventDefault();


    const p =
        pointerPos(e);


    ctx.beginPath();


    ctx.moveTo(
        lastX,
        lastY
    );


    ctx.lineTo(
        p.x,
        p.y
    );


    ctx.stroke();


    lastX = p.x;

    lastY = p.y;
}


// Stop drawing
function endDraw() {

    drawing = false;
}


// Mouse events
canvas.addEventListener(
    'mousedown',
    startDraw
);

canvas.addEventListener(
    'mousemove',
    moveDraw
);

window.addEventListener(
    'mouseup',
    endDraw
);


// Touch events
canvas.addEventListener(
    'touchstart',
    startDraw,
    {
        passive: false
    }
);

canvas.addEventListener(
    'touchmove',
    moveDraw,
    {
        passive: false
    }
);

canvas.addEventListener(
    'touchend',
    endDraw
);


// --------------------------------------------------
// Erase button
// --------------------------------------------------

eraseBtn.addEventListener(
    'click',
    () => {

        ctx.fillStyle = '#000';


        ctx.fillRect(
            0,
            0,
            canvas.width,
            canvas.height
        );


        bigDigit.className =
            'big-digit placeholder';


        bigDigit.textContent =
            'draw something';


        confidenceNote.textContent =
            'Its confidence for each digit will show up here.';


        for (let d = 0; d < 10; d++) {

            document
                .getElementById(`fill-${d}`)
                .style.width = '0%';


            document
                .getElementById(`fill-${d}`)
                .classList.remove('top');


            document
                .getElementById(`pct-${d}`)
                .textContent = '0%';
        }
    }
);


// --------------------------------------------------
// Guess button
// --------------------------------------------------

guessBtn.addEventListener(
    'click',
    () => {

        if (!trainedModel) return;


        const small =
            document.createElement('canvas');


        small.width = 28;

        small.height = 28;


        const sctx =
            small.getContext('2d');


        sctx.drawImage(
            canvas,
            0,
            0,
            28,
            28
        );


        const data =
            sctx.getImageData(
                0,
                0,
                28,
                28
            ).data;


        const input =
            new Float32Array(
                28 * 28
            );


        for (
            let i = 0;
            i < 28 * 28;
            i++
        ) {

            input[i] =
                data[i * 4] / 255;
        }


        const tensor =
            tf.tensor4d(
                input,
                [1, 28, 28, 1]
            );


        const preds =
            trainedModel.predict(tensor);


        preds.data().then(
            arr => {

                const probs =
                    Array.from(arr);


                let top = 0;


                probs.forEach(
                    (p, i) => {

                        if (
                            p > probs[top]
                        ) {
                            top = i;
                        }
                    }
                );


                bigDigit.className =
                    'big-digit';


                bigDigit.textContent =
                    top;


                confidenceNote.textContent =
                    `${(probs[top] * 100).toFixed(1)}% confident it's a ${top}.`;


                for (
                    let d = 0;
                    d < 10;
                    d++
                ) {

                    const pct =
                        probs[d] * 100;


                    const fill =
                        document.getElementById(
                            `fill-${d}`
                        );


                    fill.style.width =
                        pct + '%';


                    fill.classList.toggle(
                        'top',
                        d === top
                    );


                    document
                        .getElementById(
                            `pct-${d}`
                        )
                        .textContent =
                        pct.toFixed(0) + '%';
                }
            }
        );


        tf.dispose(tensor);

        tf.dispose(preds);
    }
);


// Start the application
main();
