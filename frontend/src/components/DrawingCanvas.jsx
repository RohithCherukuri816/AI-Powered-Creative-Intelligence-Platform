import React, { useRef, useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { Brush, Eraser, RotateCcw, Trash2, Palette } from 'lucide-react';

const DrawingCanvas = ({ onSketchChange }) => {
    const canvasRef = useRef(null);
    const [isDrawing, setIsDrawing] = useState(false);
    const [brushSize, setBrushSize] = useState(5);
    const [brushColor, setBrushColor] = useState('#6B46C1');
    const [tool, setTool] = useState('brush');
    const [history, setHistory] = useState([]);

    const colors = ['#6B46C1', '#EC4899', '#3B82F6', '#10B981', '#F59E0B', '#EF4444'];

    useEffect(() => {
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');

        // Set canvas size
        canvas.width = 600;
        canvas.height = 400;

        // Set initial canvas style
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Save initial state
        setHistory([ctx.getImageData(0, 0, canvas.width, canvas.height)]);
    }, []);

    const startDrawing = (e) => {
        setIsDrawing(true);
        const canvas = canvasRef.current;
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        const ctx = canvas.getContext('2d');
        ctx.beginPath();
        ctx.moveTo(x, y);
    };

    const draw = (e) => {
        if (!isDrawing) return;

        const canvas = canvasRef.current;
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        const ctx = canvas.getContext('2d');

        if (tool === 'brush') {
            ctx.globalCompositeOperation = 'source-over';
            ctx.strokeStyle = brushColor;
            ctx.lineWidth = brushSize;
        } else if (tool === 'eraser') {
            ctx.globalCompositeOperation = 'destination-out';
            ctx.lineWidth = brushSize * 2;
        }

        ctx.lineTo(x, y);
        ctx.stroke();
    };

    const stopDrawing = () => {
        if (!isDrawing) return;
        setIsDrawing(false);

        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');

        // Save state to history
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        setHistory(prev => [...prev.slice(-9), imageData]); // Keep last 10 states

        // Convert canvas to base64 and notify parent
        const dataURL = canvas.toDataURL('image/png');
        onSketchChange(dataURL);
    };

    const undo = () => {
        if (history.length > 1) {
            const newHistory = history.slice(0, -1);
            setHistory(newHistory);

            const canvas = canvasRef.current;
            const ctx = canvas.getContext('2d');
            ctx.putImageData(newHistory[newHistory.length - 1], 0, 0);

            const dataURL = canvas.toDataURL('image/png');
            onSketchChange(dataURL);
        }
    };

    const clearCanvas = () => {
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        setHistory([imageData]);
        onSketchChange(canvas.toDataURL('image/png'));
    };

    return (
        <motion.div
            className="glass-card rounded-2xl p-6 max-w-4xl mx-auto"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
        >
            <h2 className="text-2xl font-bold text-gray-800 mb-4 text-center">
                🎨 Draw Your Idea
            </h2>

            {/* Toolbar */}
            <div className="flex flex-wrap items-center justify-center gap-4 mb-6 p-4 bg-white/50 rounded-xl">
                {/* Tool Selection */}
                <div className="flex gap-2">
                    <motion.button
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        onClick={() => setTool('brush')}
                        className={`p-3 rounded-lg transition-all ${tool === 'brush'
                                ? 'bg-pastel-purple text-white shadow-lg'
                                : 'bg-white/80 text-gray-600 hover:bg-pastel-purple/20'
                            }`}
                    >
                        <Brush size={20} />
                    </motion.button>

                    <motion.button
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        onClick={() => setTool('eraser')}
                        className={`p-3 rounded-lg transition-all ${tool === 'eraser'
                                ? 'bg-pastel-purple text-white shadow-lg'
                                : 'bg-white/80 text-gray-600 hover:bg-pastel-purple/20'
                            }`}
                    >
                        <Eraser size={20} />
                    </motion.button>
                </div>

                {/* Brush Size */}
                <div className="flex items-center gap-2">
                    <span className="text-sm font-medium text-gray-600">Size:</span>
                    <input
                        type="range"
                        min="1"
                        max="20"
                        value={brushSize}
                        onChange={(e) => setBrushSize(parseInt(e.target.value))}
                        className="w-20 accent-pastel-purple"
                    />
                    <span className="text-sm text-gray-600 w-6">{brushSize}</span>
                </div>

                {/* Color Palette */}
                <div className="flex items-center gap-2">
                    <Palette size={16} className="text-gray-600" />
                    <div className="flex gap-1">
                        {colors.map((color) => (
                            <motion.button
                                key={color}
                                whileHover={{ scale: 1.1 }}
                                whileTap={{ scale: 0.9 }}
                                onClick={() => setBrushColor(color)}
                                className={`w-8 h-8 rounded-full border-2 transition-all ${brushColor === color ? 'border-gray-800 scale-110' : 'border-white'
                                    }`}
                                style={{ backgroundColor: color }}
                            />
                        ))}
                    </div>
                </div>

                {/* Actions */}
                <div className="flex gap-2">
                    <motion.button
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        onClick={undo}
                        disabled={history.length <= 1}
                        className="p-3 rounded-lg bg-white/80 text-gray-600 hover:bg-pastel-blue/20 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                    >
                        <RotateCcw size={20} />
                    </motion.button>

                    <motion.button
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        onClick={clearCanvas}
                        className="p-3 rounded-lg bg-white/80 text-gray-600 hover:bg-red-100 transition-all"
                    >
                        <Trash2 size={20} />
                    </motion.button>
                </div>
            </div>

            {/* Canvas */}
            <div className="flex justify-center">
                <motion.canvas
                    ref={canvasRef}
                    onMouseDown={startDrawing}
                    onMouseMove={draw}
                    onMouseUp={stopDrawing}
                    onMouseLeave={stopDrawing}
                    className="border-2 border-pastel-purple/30 rounded-xl cursor-crosshair shadow-lg bg-white"
                    whileHover={{ scale: 1.01 }}
                    transition={{ duration: 0.2 }}
                />
            </div>

            <p className="text-center text-gray-600 mt-4 text-sm">
                Draw your idea above, then add a style prompt below to generate your design! ✨
            </p>
        </motion.div>
    );
};

export default DrawingCanvas;