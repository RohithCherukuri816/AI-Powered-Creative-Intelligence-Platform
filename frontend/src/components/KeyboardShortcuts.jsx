import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Keyboard, X } from 'lucide-react';

const KeyboardShortcuts = () => {
    const [isVisible, setIsVisible] = useState(false);
    const [showHint, setShowHint] = useState(true);

    useEffect(() => {
        // Show hint for first-time users
        const hasSeenHint = localStorage.getItem('keyboardShortcutsHintSeen');
        if (!hasSeenHint) {
            const timer = setTimeout(() => {
                setShowHint(true);
            }, 3000);
            return () => clearTimeout(timer);
        } else {
            setShowHint(false);
        }
    }, []);

    const dismissHint = () => {
        setShowHint(false);
        localStorage.setItem('keyboardShortcutsHintSeen', 'true');
    };

    const shortcuts = [
        {
            category: 'Tools', items: [
                { key: 'B', description: 'Switch to Brush tool' },
                { key: 'E', description: 'Switch to Eraser tool' },

            ]
        },
        {
            category: 'Canvas', items: [
                { key: 'Ctrl + Z', description: 'Undo last action' },
                { key: 'Ctrl + S', description: 'Download sketch' },
                { key: 'Ctrl + C', description: 'Clear canvas' },
            ]
        },

        {
            category: 'Brush', items: [
                { key: '[', description: 'Decrease brush size' },
                { key: ']', description: 'Increase brush size' },
            ]
        },
    ];

    return (
        <>
            {/* Floating Hint */}
            <AnimatePresence>
                {showHint && (
                    <motion.div
                        className="fixed bottom-6 right-6 z-50"
                        initial={{ opacity: 0, y: 50, scale: 0.9 }}
                        animate={{ opacity: 1, y: 0, scale: 1 }}
                        exit={{ opacity: 0, y: 50, scale: 0.9 }}
                        transition={{ duration: 0.3 }}
                    >
                        <div className="bg-gradient-to-r from-orange-500 to-rose-500 text-white p-4 rounded-xl shadow-lg max-w-xs">
                            <div className="flex items-start gap-3">
                                <Keyboard size={20} className="flex-shrink-0 mt-0.5" />
                                <div className="flex-1">
                                    <p className="font-medium text-sm">Pro Tip!</p>
                                    <p className="text-xs opacity-90 mt-1">
                                        Use keyboard shortcuts for faster drawing. Press the keyboard icon to see all shortcuts.
                                    </p>
                                </div>
                                <button
                                    onClick={dismissHint}
                                    className="text-white/80 hover:text-white transition-colors"
                                >
                                    <X size={16} />
                                </button>
                            </div>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Keyboard Shortcuts Button */}
            <motion.button
                onClick={() => setIsVisible(true)}
                className="fixed bottom-6 left-6 z-40 bg-white/90 backdrop-blur-sm border border-orange-200 text-gray-700 p-3 rounded-full shadow-lg hover:shadow-xl transition-all duration-300 group"
                whileHover={{ scale: 1.05, y: -2 }}
                whileTap={{ scale: 0.95 }}
                title="Keyboard Shortcuts"
            >
                <Keyboard size={20} className="group-hover:text-orange-600 transition-colors" />
            </motion.button>

            {/* Shortcuts Modal */}
            <AnimatePresence>
                {isVisible && (
                    <motion.div
                        className="fixed inset-0 z-50 flex items-center justify-center p-4"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                    >
                        {/* Backdrop */}
                        <motion.div
                            className="absolute inset-0 bg-black/50 backdrop-blur-sm"
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            exit={{ opacity: 0 }}
                            onClick={() => setIsVisible(false)}
                        />

                        {/* Modal */}
                        <motion.div
                            className="relative bg-white rounded-2xl shadow-2xl max-w-2xl w-full max-h-[80vh] overflow-hidden"
                            initial={{ opacity: 0, scale: 0.9, y: 20 }}
                            animate={{ opacity: 1, scale: 1, y: 0 }}
                            exit={{ opacity: 0, scale: 0.9, y: 20 }}
                            transition={{ type: "spring", duration: 0.3 }}
                        >
                            {/* Header */}
                            <div className="bg-gradient-to-r from-orange-500 to-rose-500 text-white p-6">
                                <div className="flex items-center justify-between">
                                    <div className="flex items-center gap-3">
                                        <Keyboard size={24} />
                                        <h2 className="text-xl font-bold">Keyboard Shortcuts</h2>
                                    </div>
                                    <button
                                        onClick={() => setIsVisible(false)}
                                        className="text-white/80 hover:text-white transition-colors p-1"
                                    >
                                        <X size={20} />
                                    </button>
                                </div>
                            </div>

                            {/* Content */}
                            <div className="p-6 overflow-y-auto">
                                <div className="grid md:grid-cols-2 gap-6">
                                    {shortcuts.map((category, index) => (
                                        <motion.div
                                            key={category.category}
                                            className="space-y-3"
                                            initial={{ opacity: 0, y: 20 }}
                                            animate={{ opacity: 1, y: 0 }}
                                            transition={{ delay: index * 0.1 }}
                                        >
                                            <h3 className="font-semibold text-gray-800 text-sm uppercase tracking-wide border-b border-gray-200 pb-2">
                                                {category.category}
                                            </h3>
                                            <div className="space-y-2">
                                                {category.items.map((shortcut, i) => (
                                                    <div key={i} className="flex items-center justify-between">
                                                        <span className="text-sm text-gray-600">{shortcut.description}</span>
                                                        <kbd className="bg-gray-100 text-gray-800 px-2 py-1 rounded text-xs font-mono border border-gray-300">
                                                            {shortcut.key}
                                                        </kbd>
                                                    </div>
                                                ))}
                                            </div>
                                        </motion.div>
                                    ))}
                                </div>

                                {/* Footer tip */}
                                <motion.div
                                    className="mt-6 p-4 bg-orange-50 rounded-xl border border-orange-200"
                                    initial={{ opacity: 0 }}
                                    animate={{ opacity: 1 }}
                                    transition={{ delay: 0.4 }}
                                >
                                    <p className="text-sm text-orange-700">
                                        <strong>💡 Pro Tip:</strong> These shortcuts work when the canvas is focused.
                                        Touch and mobile users can use the toolbar buttons instead.
                                    </p>
                                </motion.div>
                            </div>
                        </motion.div>
                    </motion.div>
                )}
            </AnimatePresence>
        </>
    );
};

export default KeyboardShortcuts;