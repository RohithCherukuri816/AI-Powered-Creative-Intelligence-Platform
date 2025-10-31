import React from 'react';
import { motion } from 'framer-motion';
import { Eye, Loader, CheckCircle, AlertCircle } from 'lucide-react';

const RecognitionStatus = ({ isLoading, recognizedLabel, confidence, recognitionSuccess }) => {
    if (!isLoading && !recognizedLabel) return null;

    return (
        <motion.div
            className="fixed bottom-6 right-6 z-50"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
        >
            <div className="bg-white/90 backdrop-blur-md border border-gray-200 rounded-xl p-4 shadow-lg max-w-xs">
                {isLoading ? (
                    <div className="flex items-center gap-3">
                        <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center">
                            <Loader size={16} className="text-blue-600 animate-spin" />
                        </div>
                        <div>
                            <div className="text-sm font-medium text-gray-800">
                                Analyzing doodle...
                            </div>
                            <div className="text-xs text-gray-500">
                                AI recognition in progress
                            </div>
                        </div>
                    </div>
                ) : recognizedLabel ? (
                    <div className="flex items-center gap-3">
                        <div className={`w-8 h-8 rounded-full flex items-center justify-center ${recognitionSuccess
                                ? 'bg-green-100 text-green-600'
                                : 'bg-yellow-100 text-yellow-600'
                            }`}>
                            {recognitionSuccess ? <CheckCircle size={16} /> : <AlertCircle size={16} />}
                        </div>
                        <div>
                            <div className="text-sm font-medium text-gray-800 capitalize">
                                {recognizedLabel.replace(/-/g, ' ')}
                            </div>
                            <div className="text-xs text-gray-500">
                                {confidence ? `${Math.round(confidence * 100)}% confidence` : 'Detected'}
                            </div>
                        </div>
                    </div>
                ) : null}
            </div>
        </motion.div>
    );
};

export default RecognitionStatus;