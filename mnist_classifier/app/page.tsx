'use client';

import Image from "next/image";

import React, { useRef, useState, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as ort from 'onnxruntime-web';

const HandwritingDetection = () => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [model, setModel] = useState<ort.InferenceSession | null>(null);
  const [prediction, setPrediction] = useState<number | null>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  
  // Load ONNX model when component mounts
  React.useEffect(() => {
    const loadModel = async () => {
      const session = await ort.InferenceSession.create('model.onnx');  // Replace with your model path
      setModel(session);
    };
    
    loadModel();
  }, []);
  
  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.beginPath();
    ctx.moveTo(e.nativeEvent.offsetX, e.nativeEvent.offsetY);
    setIsDrawing(true);
  };

  const handleMouseMove = (e: MouseEvent) => {
    if (!isDrawing) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.strokeStyle = 'white';
    ctx.lineTo(e.offsetX, e.offsetY);
    ctx.stroke();
  };

  const handleMouseUp = () => {
    setIsDrawing(false);
  };

  // Function to clear the canvas
  const handleClearCanvas = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.addEventListener('mousemove', handleMouseMove);
    return () => {
      canvas.removeEventListener('mousemove', handleMouseMove);
    };
  }, [isDrawing]);
  
  // Convert the canvas drawing to a tensor
  const handlePredict = async () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // Get the canvas image data and resize it to 28x28 (MNIST size)
    const resizedImageData = resizeImageData(canvas, 28, 28);
    if (!resizedImageData) return;
    
    // Convert image data to grayscale and normalize it
    const tensor = tf.tidy(() => {
      const imageDataArray = new Float32Array(resizedImageData);
      const tensor = tf.tensor(imageDataArray, [28, 28, 1], 'float32');
      return tensor.div(255.0).reshape([1, 1, 28, 28]);  // Normalizing and reshaping for MNIST model
    });
    
    // Run the model prediction
    if (model) {
      const input = new ort.Tensor('float32', tensor.dataSync(), [1, 1, 28, 28]);
      const output = await model.run({ input });
      const outputTensor = output[Object.keys(output)[0]];
      if (!outputTensor) { return; }
      
      // Convert output tensor data to a JavaScript array
      const outputData = Array.from(outputTensor.data as Float32Array);
      
      // Get the argmax of the output tensor
      const predictedValue = outputData.reduce((maxIndex, currentValue, currentIndex, array) => {
        return currentValue > array[maxIndex] ? currentIndex : maxIndex;
      }, 0);
      
      setPrediction(predictedValue);
      tensor.dispose();
    }
  };
  
  // Resize image data (canvas) to 28x28 pixels
  const resizeImageData = (canvas: HTMLCanvasElement, width: number, height: number) => {
    const resizeCanvas = document.createElement('canvas');
    resizeCanvas.width = width;
    resizeCanvas.height = height;
    const ctx = resizeCanvas.getContext('2d');
    if (!ctx) return;
    ctx.drawImage(canvas, 0, 0, width, height);
    const resizedImageData = ctx.getImageData(0, 0, width, height);
    // Convert the resized image data to grayscale
    const grayscaleData = new Uint8ClampedArray(resizedImageData.data.length / 4);
    for (let i = 0; i < resizedImageData.data.length; i += 4) {
      const r = resizedImageData.data[i];
      const g = resizedImageData.data[i + 1];
      const b = resizedImageData.data[i + 2];
      const gray = 0.299 * r + 0.587 * g + 0.114 * b;
      grayscaleData[i / 4] = gray;
    }
    
    return grayscaleData;
  };
  
  return (
    <div>
    <h1>Handwriting Detection</h1>
    <canvas
    ref={canvasRef}
    width={280}
    height={280}
    style={{ border: '1px solid black' }}
    onMouseDown={handleMouseDown}
    onMouseUp={handleMouseUp}
    />
    <div>
    <button onClick={handlePredict} style={{ marginRight: '10px' }}>Predict</button>
    <button onClick={handleClearCanvas}>Clear</button>
    </div>
    {prediction !== null && <h2>Prediction: {prediction}</h2>}
    </div>
  );
};


export default function Home() {
  return (
    <div className="grid grid-rows-[20px_1fr_20px] items-center justify-items-center min-h-screen p-8 pb-20 gap-16 sm:p-20 font-[family-name:var(--font-geist-sans)]">
    <main className="flex flex-col gap-8 row-start-2 items-center sm:items-start">
    <Image
    className="dark:invert"
    src="https://nextjs.org/icons/next.svg"
    alt="Next.js logo"
    width={180}
    height={38}
    priority
    />
    <ol className="list-inside list-decimal text-sm text-center sm:text-left font-[family-name:var(--font-geist-mono)]">
    <li className="mb-2">
    Get started by editing{" "}
    <code className="bg-black/[.05] dark:bg-white/[.06] px-1 py-0.5 rounded font-semibold">
    app/page.tsx
    </code>
    .
    </li>
    <li>Save and see your changes instantly.</li>
    </ol>
    
    <HandwritingDetection />
    
    <div className="flex gap-4 items-center flex-col sm:flex-row">
    <a
    className="rounded-full border border-solid border-transparent transition-colors flex items-center justify-center bg-foreground text-background gap-2 hover:bg-[#383838] dark:hover:bg-[#ccc] text-sm sm:text-base h-10 sm:h-12 px-4 sm:px-5"
    href="https://vercel.com/new?utm_source=create-next-app&utm_medium=appdir-template-tw&utm_campaign=create-next-app"
    target="_blank"
    rel="noopener noreferrer"
    >
    <Image
    className="dark:invert"
    src="https://nextjs.org/icons/vercel.svg"
    alt="Vercel logomark"
    width={20}
    height={20}
    />
    Deploy now
    </a>
    <a
    className="rounded-full border border-solid border-black/[.08] dark:border-white/[.145] transition-colors flex items-center justify-center hover:bg-[#f2f2f2] dark:hover:bg-[#1a1a1a] hover:border-transparent text-sm sm:text-base h-10 sm:h-12 px-4 sm:px-5 sm:min-w-44"
    href="https://nextjs.org/docs?utm_source=create-next-app&utm_medium=appdir-template-tw&utm_campaign=create-next-app"
    target="_blank"
    rel="noopener noreferrer"
    >
    Read our docs
    </a>
    </div>
    </main>
    <footer className="row-start-3 flex gap-6 flex-wrap items-center justify-center">
    <a
    className="flex items-center gap-2 hover:underline hover:underline-offset-4"
    href="https://nextjs.org/learn?utm_source=create-next-app&utm_medium=appdir-template-tw&utm_campaign=create-next-app"
    target="_blank"
    rel="noopener noreferrer"
    >
    <Image
    aria-hidden
    src="https://nextjs.org/icons/file.svg"
    alt="File icon"
    width={16}
    height={16}
    />
    Learn
    </a>
    <a
    className="flex items-center gap-2 hover:underline hover:underline-offset-4"
    href="https://vercel.com/templates?framework=next.js&utm_source=create-next-app&utm_medium=appdir-template-tw&utm_campaign=create-next-app"
    target="_blank"
    rel="noopener noreferrer"
    >
    <Image
    aria-hidden
    src="https://nextjs.org/icons/window.svg"
    alt="Window icon"
    width={16}
    height={16}
    />
    Examples
    </a>
    <a
    className="flex items-center gap-2 hover:underline hover:underline-offset-4"
    href="https://nextjs.org?utm_source=create-next-app&utm_medium=appdir-template-tw&utm_campaign=create-next-app"
    target="_blank"
    rel="noopener noreferrer"
    >
    <Image
    aria-hidden
    src="https://nextjs.org/icons/globe.svg"
    alt="Globe icon"
    width={16}
    height={16}
    />
    Go to nextjs.org →
    </a>
    </footer>
    </div>
  );
}
