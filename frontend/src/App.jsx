import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Sparkles, Palette, Zap, Heart } from 'lucide-react';
import DrawingCanvas from './components/DrawingCanvas';
import PromptInput from './components/PromptInput';
import GeneratedImageCard from './components/GeneratedImageCard';
import { api } from './api';

function App() {
    const [sketchData, setSketchData] = useState(null);
    const [generatedImage, setGeneratedImage] = useState(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);

    const handleSketchChange = (dataURL) => {
        setSketchData(dataURL);
        setError(null);
    };

    const handleGenerate = async (prompt) => {
        if (!sketchData) {
            setError('Please draw something first!');
            return;
        }

        setIsLoading(true);
        setError(null);

        try {
            const result = await api.generateDesign(prompt, sketchData);
            setGeneratedImage(result.image_url);
        } catch (error) {
            console.error('Generation failed:', error);
            setError('Failed to generate design. Please try again!');
        } finally {
            setIsLoading(false);
        }
    };

    const handleTryAgain = () => {
        setGeneratedImage(null);
        setError(null);
    };

    return (
        <div className="min-h-screen">
            {/* Hero Section */}
            <motion.section
                className="relative overflow-hidden py-20 px-4"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 1 }}
            >
                {/* Floating Elements */}
                <div className="absolute inset-0 overflow-hidden pointer-events-none">
                    <motion.div
                        className="absolute top-20 left-10 w-20 h-20 bg-pastel-purple/20 rounded-full"
                        animate={{ y: [0, -20, 0], rotate: [0, 180, 360] }}
                        transition={{ duration: 8, repeat: Infinity, ease: "easeInOut" }}
                    />
                    <motion.div
                        className="absolute top-40 right-20 w-16 h-16 bg-pastel-pink/20 rounded-full"
                        animate={{ y: [0, 20, 0], x: [0, -10, 0] }}
                        transition={{ duration: 6, repeat: Infinity, ease: "easeInOut" }}
                    />
                    <motion.div
                        className="absolute bottom-40 left-1/4 w-12 h-12 bg-pastel-blue/20 rounded-full"
                        animate={{ y: [0, -15, 0], rotate: [0, -180, -360] }}
                        transition={{ duration: 7, repeat: Infinity, ease: "easeInOut" }}
                    />
                </div>

                <div className="max-w-6xl mx-auto text-center relative z-10">
                    <motion.div
                        initial={{ opacity: 0, y: 30 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.8 }}
                    >
                        <h1 className="text-5xl md:text-7xl font-bold text-gray-800 mb-6 leading-tight">
                            <span className="bg-gradient-to-r from-pastel-purple via-pastel-pink to-pastel-blue bg-clip-text text-transparent">
                                ✨ Transform Your Doodles
                            </span>
                            <br />
                            <span className="text-gray-700">
                                into Stunning Designs
                            </span>
                        </h1>
                    </motion.div>

                    <motion.p
                        className="text-xl md:text-2xl text-gray-600 mb-8 max-w-3xl mx-auto leading-relaxed"
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.8, delay: 0.2 }}
                    >
                        Sketch. Prompt. Generate. Bring your ideas to life in seconds with the power of AI.
                    </motion.p>

                    <motion.div
                        className="flex flex-wrap justify-center gap-8 mb-12"
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.8, delay: 0.4 }}
                    >
                        <div className="flex items-center gap-3 text-gray-700">
                            <div className="p-3 bg-pastel-purple/20 rounded-full">
                                <Palette size={24} className="text-pastel-purple" />
                            </div>
                            <span className="font-medium">Draw Anything</span>
                        </div>
                        <div className="flex items-center gap-3 text-gray-700">
                            <div className="p-3 bg-pastel-pink/20 rounded-full">
                                <Sparkles size={24} className="text-pastel-pink" />
                            </div>
                            <span className="font-medium">AI-Powered</span>
                        </div>
                        <div className="flex items-center gap-3 text-gray-700">
                            <div className="p-3 bg-pastel-blue/20 rounded-full">
                                <Zap size={24} className="text-pastel-blue" />
                            </div>
                            <span className="font-medium">Instant Results</span>
                        </div>
                    </motion.div>
                </div>
            </motion.section>

            {/* Main Content */}
            <div className="max-w-7xl mx-auto px-4 pb-20 space-y-8">
                {/* Error Message */}
                {error && (
                    <motion.div
                        className="max-w-4xl mx-auto p-4 bg-red-100 border border-red-300 text-red-700 rounded-xl text-center"
                        initial={{ opacity: 0, y: -20 }}
                        animate={{ opacity: 1, y: 0 }}
                    >
                        {error}
                    </motion.div>
                )}

                {/* Drawing Canvas */}
                <DrawingCanvas onSketchChange={handleSketchChange} />

                {/* Prompt Input */}
                <PromptInput onGenerate={handleGenerate} isLoading={isLoading} />

                {/* Generated Image */}
                {generatedImage && (
                    <GeneratedImageCard
                        imageUrl={generatedImage}
                        onTryAgain={handleTryAgain}
                    />
                )}
            </div>

            {/* Footer */}
            <motion.footer
                className="py-12 text-center"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.8, delay: 1 }}
            >
                <div className="max-w-4xl mx-auto px-4">
                    <motion.div
                        className="flex items-center justify-center gap-2 text-gray-600 mb-4"
                        whileHover={{ scale: 1.05 }}
                    >
                        <Heart size={20} className="text-pastel-pink fill-current" />
                        <span className="text-lg font-medium">
                            Made by Rohith Cherukuri
                        </span>
                        <Heart size={20} className="text-pastel-pink fill-current" />
                    </motion.div>

                    <p className="text-gray-500 text-sm">
                        Transforming creativity with AI • Built with React, Tailwind & FastAPI
                    </p>
                </div>
            </motion.footer>
        </div>
    );
}

export default App;