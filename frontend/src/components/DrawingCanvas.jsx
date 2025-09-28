import React, { useRef, useEffect, useState, useCallback } from 'react';
import { motion } from 'framer-motion';
import { Brush, Eraser, RotateCcw, Trash2, Palette, Download } from 'lucide-react';

const DrawingCanvas = ({
    onSketchChange,
    canvasStyle,
    toolbarStyle
}) => {
    const canvasRef = useRef(null);
    const [isDrawing, setIsDrawing] = useState(false);
    const [brushSize, setBrushSize] = useState(5);
    const [brushColor, setBrushColor] = useState('#DC2626');
    const [tool, setTool] = useState('brush');
    const [history, setHistory] = useState([]);
    const [lastX, setLastX] = useState(0);
    const [lastY, setLastY] = useState(0);

    const colors = [
        '#DC2626', '#EA580C', '#D97706', '#65A30D', '#059669',
        '#2563EB', '#7C3AED', '#C026D3', '#000000', '#4B5563'
    ];

    // Default styles
    const defaultCanvasStyle = {
        border: '2px solid #ea580c',
        borderRadius: '12px',
        background: '#ffffff',
        cursor: 'crosshair'
    };

    const defaultToolbarStyle = {
        background: 'rgba(255, 255, 255, 0.95)',
        border: '1px solid #e5e7eb',
        borderRadius: '12px',
        backdropFilter: 'blur(10px)'
    };

    const mergedCanvasStyle = { ...defaultCanvasStyle, ...canvasStyle };
    const mergedToolbarStyle = { ...defaultToolbarStyle, ...toolbarStyle };

    // Keyboard shortcuts
    const handleKeyPress = useCallback((e) => {
        // Prevent default browser shortcuts
        if (e.ctrlKey || e.metaKey) {
            switch (e.key.toLowerCase()) {
                case 'z':
                    e.preventDefault();
                    if (e.shiftKey) {
                        // Redo functionality could be added here
                    } else {
                        undo();
                    }
                    break;
                case 's':
                    e.preventDefault();
                    downloadSketch();
                    break;
                case 'c':
                    e.preventDefault();
                    clearCanvas();
                    break;
                default:
                    break;
            }
        }

        // Tool shortcuts (without Ctrl)
        if (!e.ctrlKey && !e.metaKey && !e.altKey) {
            switch (e.key.toLowerCase()) {
                case 'b':
                    setTool('brush');
                    break;
                case 'e':
                    setTool('eraser');
                    break;
                case '[':
                    setBrushSize(prev => Math.max(1, prev - 1));
                    break;
                case ']':
                    setBrushSize(prev => Math.min(20, prev + 1));
                    break;
                default:
                    break;
            }
        }
    }, []);

    useEffect(() => {
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');

        // Set canvas size
        canvas.width = 600;
        canvas.height = 400;

        // Set initial canvas style
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Save initial state
        setHistory([ctx.getImageData(0, 0, canvas.width, canvas.height)]);

        // Add keyboard event listeners
        window.addEventListener('keydown', handleKeyPress);

        return () => {
            window.removeEventListener('keydown', handleKeyPress);
        };
    }, [handleKeyPress]);

    const startDrawing = (e) => {
        setIsDrawing(true);
        const { x, y } = getCanvasCoordinates(e.clientX, e.clientY);

        const ctx = canvasRef.current.getContext('2d');

        if (tool === 'eraser') {
            ctx.globalCompositeOperation = 'destination-out';
            ctx.strokeStyle = 'rgba(0,0,0,1)';
        } else {
            ctx.globalCompositeOperation = 'source-over';
            ctx.strokeStyle = brushColor;
        }

        ctx.lineWidth = tool === 'eraser' ? brushSize * 2 : brushSize;
        ctx.beginPath();
        ctx.moveTo(x, y);

        setLastX(x);
        setLastY(y);
    };

    const draw = (e) => {
        if (!isDrawing) return;

        const { x, y } = getCanvasCoordinates(e.clientX, e.clientY);
        const ctx = canvasRef.current.getContext('2d');

        // Draw smooth line
        ctx.lineTo(x, y);
        ctx.stroke();

        setLastX(x);
        setLastY(y);
    };

    const stopDrawing = () => {
        if (!isDrawing) return;
        setIsDrawing(false);

        const ctx = canvasRef.current.getContext('2d');
        ctx.closePath();

        // Save state to history
        const imageData = ctx.getImageData(0, 0, canvasRef.current.width, canvasRef.current.height);
        setHistory(prev => [...prev.slice(-9), imageData]);

        // Convert canvas to base64 and notify parent
        const dataURL = canvasRef.current.toDataURL('image/png');
        onSketchChange(dataURL);
    };

    // Touch event handlers
    const handleTouchStart = (e) => {
        e.preventDefault();
        const touch = e.touches[0];

        setIsDrawing(true);
        const { x, y } = getCanvasCoordinates(touch.clientX, touch.clientY);

        const ctx = canvasRef.current.getContext('2d');

        if (tool === 'eraser') {
            ctx.globalCompositeOperation = 'destination-out';
            ctx.strokeStyle = 'rgba(0,0,0,1)';
        } else {
            ctx.globalCompositeOperation = 'source-over';
            ctx.strokeStyle = brushColor;
        }

        ctx.lineWidth = tool === 'eraser' ? brushSize * 2 : brushSize;
        ctx.beginPath();
        ctx.moveTo(x, y);

        setLastX(x);
        setLastY(y);
    };

    const handleTouchMove = (e) => {
        e.preventDefault();
        const touch = e.touches[0];

        if (!isDrawing) return;

        const { x, y } = getCanvasCoordinates(touch.clientX, touch.clientY);
        const ctx = canvasRef.current.getContext('2d');

        // Draw smooth line
        ctx.lineTo(x, y);
        ctx.stroke();

        setLastX(x);
        setLastY(y);
    };

    const handleTouchEnd = (e) => {
        e.preventDefault();

        if (!isDrawing) return;
        setIsDrawing(false);

        const ctx = canvasRef.current.getContext('2d');
        ctx.closePath();

        // Save state to history
        const imageData = ctx.getImageData(0, 0, canvasRef.current.width, canvasRef.current.height);
        setHistory(prev => [...prev.slice(-9), imageData]);

        // Convert canvas to base64 and notify parent
        const dataURL = canvasRef.current.toDataURL('image/png');
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
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        setHistory([imageData]);

        const dataURL = canvas.toDataURL('image/png');
        onSketchChange(dataURL);
    };

    const downloadSketch = () => {
        const canvas = canvasRef.current;
        const link = document.createElement('a');
        link.download = `sketch-${Date.now()}.png`;
        link.href = canvas.toDataURL('image/png');
        link.click();
    };

    const getCanvasCoordinates = (clientX, clientY) => {
        const canvas = canvasRef.current;
        const rect = canvas.getBoundingClientRect();

        const x = clientX - rect.left;
        const y = clientY - rect.top;

        return { x, y };
    };

    return (
        <motion.div
            className="rounded-xl p-4 max-w-4xl mx-auto bg-white border border-gray-200 shadow-sm"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
        >
            {/* Simple Header */}
            <div className="text-center mb-4">
                <h2 className="text-lg font-semibold text-gray-800">Draw Your Idea</h2>
            </div>

            {/* Toolbar - Simplified like the image */}
            <div className="flex flex-wrap items-center justify-between gap-3 mb-4 p-3 bg-gray-50 rounded-lg border border-gray-200">
                {/* Left - Tools */}
                <div className="flex items-center gap-2">
                    <span className="text-sm font-medium text-gray-700">Tools:</span>
                    <div className="flex gap-1">
                        <button
                            onClick={() => setTool('brush')}
                            className={`px-3 py-2 rounded text-sm font-medium transition-colors ${tool === 'brush'
                                ? 'bg-blue-500 text-white'
                                : 'bg-white text-gray-700 border border-gray-300 hover:bg-gray-50'
                                }`}
                        >
                            <Brush size={16} />
                        </button>

                        <button
                            onClick={() => setTool('eraser')}
                            className={`px-3 py-2 rounded text-sm font-medium transition-colors ${tool === 'eraser'
                                ? 'bg-red-500 text-white'
                                : 'bg-white text-gray-700 border border-gray-300 hover:bg-gray-50'
                                }`}
                        >
                            <Eraser size={16} />
                        </button>
                    </div>
                </div>

                {/* Middle - Brush Size */}
                <div className="flex items-center gap-2">
                    <span className="text-sm font-medium text-gray-700">Size:</span>
                    <div className="flex items-center gap-2 bg-white rounded border border-gray-300 px-2 py-1">
                        <input
                            type="range"
                            min="1"
                            max="20"
                            value={brushSize}
                            onChange={(e) => setBrushSize(parseInt(e.target.value))}
                            className="w-16 accent-blue-500"
                        />
                        <span className="text-xs font-medium text-gray-600 w-4">{brushSize}</span>
                    </div>
                </div>

                {/* Color Palette */}
                <div className="flex items-center gap-2">
                    <Palette size={14} className="text-gray-600" />
                    <div className="flex gap-1">
                        {colors.slice(0, 6).map((color) => (
                            <button
                                key={color}
                                onClick={() => setBrushColor(color)}
                                className={`w-6 h-6 rounded border transition-transform ${brushColor === color ? 'border-gray-800 scale-110' : 'border-gray-300'
                                    }`}
                                style={{ backgroundColor: color }}
                            />
                        ))}
                    </div>
                </div>

                {/* Right - Actions */}
                <div className="flex items-center gap-1">
                    <button
                        onClick={undo}
                        disabled={history.length <= 1}
                        className="p-2 rounded text-gray-600 hover:bg-gray-100 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                        title="Undo (Ctrl+Z)"
                    >
                        <RotateCcw size={16} />
                    </button>

                    <button
                        onClick={downloadSketch}
                        className="p-2 rounded text-gray-600 hover:bg-gray-100 transition-colors"
                        title="Download (Ctrl+S)"
                    >
                        <Download size={16} />
                    </button>

                    <button
                        onClick={clearCanvas}
                        className="p-2 rounded text-red-600 hover:bg-red-50 transition-colors"
                        title="Clear (Ctrl+C)"
                    >
                        <Trash2 size={16} />
                    </button>
                </div>
            </div>

            {/* Canvas Container */}
            <div className="flex justify-center">
                <canvas
                    ref={canvasRef}
                    onMouseDown={startDrawing}
                    onMouseMove={draw}
                    onMouseUp={stopDrawing}
                    onMouseLeave={stopDrawing}
                    onTouchStart={handleTouchStart}
                    onTouchMove={handleTouchMove}
                    onTouchEnd={handleTouchEnd}
                    className="touch-none select-none"
                    style={mergedCanvasStyle}
                />
            </div>

            {/* Enhanced Instructions */}
            <div className="mt-4 space-y-2">
                <div className="flex justify-center gap-4 text-xs text-gray-500">
                    <div className="flex items-center gap-1">
                        <div className="w-2 h-2 rounded-full bg-blue-500"></div>
                        <span>Select brush size and color</span>
                    </div>
                    <div className="flex items-center gap-1">
                        <div className="w-2 h-2 rounded-full bg-green-500"></div>
                        <span>Draw your idea freely</span>
                    </div>
                    <div className="flex items-center gap-1">
                        <div className="w-2 h-2 rounded-full bg-orange-500"></div>
                        <span>Use undo if needed</span>
                    </div>
                </div>

                {/* Keyboard Shortcuts */}
                <div className="text-center">
                    <details className="inline-block">
                        <summary className="text-xs text-gray-400 cursor-pointer hover:text-gray-600 transition-colors">
                            ⌨️ Keyboard Shortcuts
                        </summary>
                        <div className="mt-2 p-3 bg-gray-50 rounded-lg text-xs text-gray-600 grid grid-cols-2 md:grid-cols-3 gap-2 text-left">
                            <div><kbd className="bg-white px-1 rounded border">B</kbd> Brush</div>
                            <div><kbd className="bg-white px-1 rounded border">E</kbd> Eraser</div>
                            <div><kbd className="bg-white px-1 rounded border">Ctrl+Z</kbd> Undo</div>
                            <div><kbd className="bg-white px-1 rounded border">Ctrl+S</kbd> Save</div>
                            <div><kbd className="bg-white px-1 rounded border">Ctrl+C</kbd> Clear</div>
                            <div><kbd className="bg-white px-1 rounded border">[/]</kbd> Brush Size</div>
                        </div>
                    </details>
                </div>
            </div>
        </motion.div>
    );
};

export default DrawingCanvas;