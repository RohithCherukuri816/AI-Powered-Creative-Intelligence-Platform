import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Brush, Sparkles, Zap, Heart, Github, Download, Share2, Palette, Coffee } from 'lucide-react';
import DrawingCanvas from './components/DrawingCanvas';
import PromptInput from './components/PromptInput';
import GeneratedImageCard from './components/GeneratedImageCard';
import LoadingSpinner from './components/LoadingSpinner';
import KeyboardShortcuts from './components/KeyboardShortcuts';
import TouchGestures from './components/TouchGestures';
import { api, toAbsoluteUrl } from './api';

function App() {
    const [sketchData, setSketchData] = useState(null);
    const [generatedImage, setGeneratedImage] = useState(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);
    const [activeStep, setActiveStep] = useState(1);
    const [hasDrawing, setHasDrawing] = useState(false);

    const handleSketchChange = (dataURL) => {
        setSketchData(dataURL);
        setError(null);

        // Check if the canvas has actual drawing content
        checkForDrawingContent(dataURL);
    };

    const checkForDrawingContent = (dataURL) => {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        const img = new Image();

        img.onload = function () {
            canvas.width = img.width;
            canvas.height = img.height;
            ctx.drawImage(img, 0, 0);

            const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
            const data = imageData.data;

            let drawingDetected = false;
            // Check if there are any non-white pixels (indicating a drawing)
            for (let i = 0; i < data.length; i += 4) {
                // If pixel is not white (accounting for anti-aliasing and grid lines)
                if (data[i] < 240 || data[i + 1] < 240 || data[i + 2] < 240) {
                    drawingDetected = true;
                    break;
                }
            }

            setHasDrawing(drawingDetected);
            setActiveStep(drawingDetected ? 2 : 1);
        };

        img.src = dataURL;
    };

    const handleGenerate = async (prompt) => {
        if (!sketchData || !hasDrawing) {
            setError('Please draw something first!');
            return;
        }

        setIsLoading(true);
        setError(null);
        setActiveStep(3);

        try {
            const result = await api.generateDesign(prompt, sketchData);
            setGeneratedImage(toAbsoluteUrl(result.image_url));
            setRecognizedLabel(result.recognized_label || null);
            setActiveStep(4);
        } catch (error) {
            console.error('Generation failed:', error);
            setError('Failed to create design. Please try again!');
            setActiveStep(2);
        } finally {
            setIsLoading(false);
        }
    };

    const [recognizedLabel, setRecognizedLabel] = useState(null);

    const handleTryAgain = () => {
        setGeneratedImage(null);
        setRecognizedLabel(null);
        setError(null);
        setActiveStep(2);
    };

    const steps = [
        { number: 1, title: 'Sketch', description: 'Draw your idea', icon: Brush },
        { number: 2, title: 'Style', description: 'Add some flair', icon: Palette },
        { number: 3, title: 'Create', description: 'Working magic', icon: Zap },
        { number: 4, title: 'Voila!', description: 'Your creation', icon: Sparkles }
    ];

    return (
        <div className="min-h-screen bg-gradient-to-br from-amber-50 via-orange-50 to-rose-50">
            {/* Custom background elements */}
            <div className="fixed inset-0 pointer-events-none overflow-hidden">
                <div className="absolute top-1/4 left-10 w-32 h-32 bg-yellow-200/20 rounded-full blur-xl"></div>
                <div className="absolute bottom-1/3 right-16 w-40 h-40 bg-orange-300/15 rounded-full blur-xl"></div>
                <div className="absolute top-1/2 left-1/2 w-48 h-48 bg-rose-400/10 rounded-full blur-2xl"></div>

                {/* Grid pattern */}
                <div className="absolute inset-0 opacity-[0.02] bg-[linear-gradient(rgba(0,0,0,0.3)_1px,transparent_1px),linear-gradient(90deg,rgba(0,0,0,0.3)_1px,transparent_1px)] bg-[size:64px_64px]"></div>
            </div>

            {/* Header */}
            <motion.header
                className="relative bg-white/70 backdrop-blur-md border-b border-orange-200/50 sticky top-0 z-50"
                initial={{ y: -50, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                transition={{ duration: 0.6 }}
            >
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                    <div className="flex justify-between items-center h-16">
                        {/* Logo with custom design */}
                        <motion.div
                            className="flex items-center gap-3 group"
                            whileHover={{ scale: 1.02 }}
                        >
                            <div className="relative">
                                <div className="w-10 h-10 bg-gradient-to-br from-amber-400 to-orange-500 rounded-lg flex items-center justify-center shadow-lg shadow-orange-200">
                                    <Brush className="text-white" size={22} />
                                </div>
                                <div className="absolute -top-1 -right-1 w-4 h-4 bg-rose-500 rounded-full border-2 border-white"></div>
                            </div>
                            <div>
                                <span className="text-xl font-bold text-gray-800 font-serif">
                                    SketchCraft
                                </span>
                                <div className="text-xs text-orange-600 font-medium -mt-1">
                                    by Rohith
                                </div>
                            </div>
                        </motion.div>

                        {/* GitHub Link with custom style */}
                        <motion.a
                            href="https://github.com/RohithCherukuri816/AI-Powered-Creative-Intelligence-Platform.git"
                            target="_blank"
                            rel="noopener noreferrer"
                            className="flex items-center gap-2 px-4 py-2 text-gray-700 hover:text-gray-900 transition-all duration-300 rounded-lg hover:bg-orange-50/50 border border-transparent hover:border-orange-200"
                            whileHover={{ scale: 1.05, y: -1 }}
                            whileTap={{ scale: 0.95 }}
                        >
                            <Github size={18} />
                            <span className="font-medium text-sm">Github</span>
                        </motion.a>
                    </div>
                </div>
            </motion.header>

            {/* Hero Section */}
            <motion.section
                className="relative py-16 px-4"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 1 }}
            >
                <div className="max-w-4xl mx-auto text-center">
                    <motion.div
                        initial={{ opacity: 0, y: 30 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.8 }}
                    >
                        <div className="inline-flex items-center gap-2 px-4 py-2 bg-amber-100/50 rounded-full border border-amber-200 text-amber-700 text-sm font-medium mb-6">
                            <Coffee size={16} />
                            Hand-crafted creativity tool
                        </div>

                        <h1 className="text-4xl md:text-6xl font-bold text-gray-800 mb-6 font-serif">
                            Draw.
                            <span className="text-orange-600 mx-2">Style.</span>
                            <span className="text-rose-600">Create.</span>
                        </h1>

                        <p className="text-lg text-gray-600 mb-8 max-w-2xl mx-auto leading-relaxed">
                            Transform your sketches into beautiful designs with our creative toolkit.
                            No AI buzzwords, just pure creative magic.
                        </p>
                    </motion.div>

                    {/* Custom Progress Steps */}
                    <motion.div
                        className="flex justify-center items-center mb-12"
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.8, delay: 0.3 }}
                    >
                        <div className="flex items-center space-x-6 md:space-x-12">
                            {steps.map((step, index) => {
                                const StepIcon = step.icon;
                                const isCompleted = step.number < activeStep;
                                const isActive = step.number === activeStep;

                                return (
                                    <React.Fragment key={step.number}>
                                        <div className="flex flex-col items-center">
                                            <motion.div
                                                className={`relative w-14 h-14 rounded-xl flex items-center justify-center border-2 transition-all duration-300 shadow-lg ${isCompleted
                                                    ? 'bg-emerald-500 border-emerald-500 text-white shadow-emerald-200'
                                                    : isActive
                                                        ? 'bg-gradient-to-br from-orange-500 to-rose-500 border-orange-500 text-white shadow-orange-200'
                                                        : 'bg-white border-gray-300 text-gray-400 shadow-gray-200'
                                                    }`}
                                                whileHover={{ scale: 1.05, y: -2 }}
                                            >
                                                {isCompleted ? (
                                                    <motion.div
                                                        initial={{ scale: 0 }}
                                                        animate={{ scale: 1 }}
                                                        transition={{ type: "spring", stiffness: 200 }}
                                                    >
                                                        ✓
                                                    </motion.div>
                                                ) : (
                                                    <StepIcon size={22} />
                                                )}

                                                {/* Step number badge */}
                                                <div className={`absolute -top-2 -right-2 w-6 h-6 rounded-full text-xs font-bold flex items-center justify-center border-2 ${isCompleted
                                                    ? 'bg-emerald-500 text-white border-white'
                                                    : isActive
                                                        ? 'bg-orange-500 text-white border-white'
                                                        : 'bg-gray-300 text-gray-600 border-white'
                                                    }`}>
                                                    {step.number}
                                                </div>
                                            </motion.div>
                                            <div className="mt-3 text-center">
                                                <div className={`text-sm font-semibold ${isActive ? 'text-orange-600' :
                                                    isCompleted ? 'text-emerald-600' : 'text-gray-500'
                                                    }`}>
                                                    {step.title}
                                                </div>
                                                <div className="text-xs text-gray-400 mt-1">
                                                    {step.description}
                                                </div>
                                            </div>
                                        </div>

                                        {index < steps.length - 1 && (
                                            <motion.div
                                                className={`w-8 h-1 rounded-full ${step.number < activeStep ? 'bg-emerald-400' : 'bg-gray-300'
                                                    }`}
                                                whileHover={{ scale: 1.1 }}
                                            />
                                        )}
                                    </React.Fragment>
                                );
                            })}
                        </div>
                    </motion.div>
                </div>
            </motion.section>

            {/* Main Content */}
            <div className="relative max-w-6xl mx-auto px-4 pb-16 space-y-10">
                {/* Error Message */}
                <AnimatePresence>
                    {error && (
                        <motion.div
                            className="max-w-4xl mx-auto p-4 bg-rose-50 border border-rose-200 text-rose-700 rounded-xl text-center shadow-lg"
                            initial={{ opacity: 0, y: -20 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -20 }}
                        >
                            <div className="flex items-center justify-center gap-2">
                                <div className="w-2 h-2 bg-rose-500 rounded-full"></div>
                                {error}
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>

                {/* Drawing Canvas */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.6 }}
                >
                    <DrawingCanvas
                        onSketchChange={handleSketchChange}
                    />
                </motion.div>

                {/* Enhanced Prompt Input - Only show if there's an actual drawing */}
                {hasDrawing && (
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.6, delay: 0.2 }}
                    >
                        <PromptInput
                            onGenerate={handleGenerate}
                            isLoading={isLoading}
                        />
                    </motion.div>
                )}

                {/* Loading State */}
                <AnimatePresence>
                    {isLoading && (
                        <motion.div
                            initial={{ opacity: 0, scale: 0.9 }}
                            animate={{ opacity: 1, scale: 1 }}
                            exit={{ opacity: 0, scale: 0.9 }}
                            className="flex justify-center"
                        >
                            <div className="bg-white/80 backdrop-blur-sm rounded-2xl p-8 max-w-md mx-auto border border-orange-200 shadow-xl">
                                <LoadingSpinner message="Crafting your design..." />
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>

                {/* Generated Image */}
                <AnimatePresence>
                    {generatedImage && !isLoading && (
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.6 }}
                        >
                            <GeneratedImageCard
                                imageUrl={generatedImage}
                                onTryAgain={handleTryAgain}
                            />
                        </motion.div>
                    )}
                </AnimatePresence>

                {/* Instruction for empty canvas */}
                {!hasDrawing && activeStep === 1 && (
                    <motion.div
                        className="max-w-4xl mx-auto text-center p-6 bg-amber-50/50 rounded-xl border border-amber-200"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ delay: 0.5 }}
                    >
                        <div className="flex items-center justify-center gap-3 text-amber-700">
                            <div className="w-8 h-8 bg-amber-100 rounded-full flex items-center justify-center">
                                <Brush size={16} />
                            </div>
                            <div className="text-left">
                                <p className="font-medium">Start drawing to continue</p>
                                <p className="text-sm text-amber-600">Draw something on the canvas above to unlock style options</p>
                            </div>
                        </div>
                    </motion.div>
                )}
            </div>

            {/* Footer */}
            <motion.footer
                className="relative py-12 bg-white/30 backdrop-blur-sm border-t border-orange-200/50 mt-20"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.8, delay: 1 }}
            >
                <div className="max-w-4xl mx-auto px-4 text-center">
                    <motion.div
                        className="flex items-center justify-center gap-3 text-gray-700 mb-4"
                        whileHover={{ scale: 1.05 }}
                    >
                        <div className="w-6 h-6 bg-gradient-to-br from-amber-400 to-orange-500 rounded-full flex items-center justify-center">
                            <Heart size={12} className="text-white fill-current" />
                        </div>
                        <span className="text-lg font-semibold font-serif">
                            Crafted by Rohith Cherukuri
                        </span>
                    </motion.div>

                    <p className="text-gray-600 text-sm mb-6 max-w-md mx-auto">
                        A creative toolkit for designers and artists who love to experiment with visual styles.
                    </p>

                    <div className="flex justify-center gap-6">
                        <motion.button
                            className="flex items-center gap-2 px-5 py-2 text-gray-600 hover:text-gray-900 transition-all duration-300 rounded-lg hover:bg-orange-50/50 border border-transparent hover:border-orange-200"
                            whileHover={{ scale: 1.05, y: -1 }}
                            whileTap={{ scale: 0.95 }}
                        >
                            <Share2 size={16} />
                            <span className="font-medium text-sm">Share Project</span>
                        </motion.button>

                        <motion.a
                            href="#"
                            className="flex items-center gap-2 px-5 py-2 text-gray-600 hover:text-gray-900 transition-all duration-300 rounded-lg hover:bg-orange-50/50 border border-transparent hover:border-orange-200"
                            whileHover={{ scale: 1.05, y: -1 }}
                            whileTap={{ scale: 0.95 }}
                        >
                            <Download size={16} />
                            <span className="font-medium text-sm">Export Kit</span>
                        </motion.a>
                    </div>
                </div>
            </motion.footer>

            {/* Keyboard Shortcuts Helper */}
            <KeyboardShortcuts />

            {/* Touch Gestures Helper */}
            <TouchGestures />
        </div>
    );
}

export default App;