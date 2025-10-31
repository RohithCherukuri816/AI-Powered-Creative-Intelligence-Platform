import React from 'react';
import { motion } from 'framer-motion';
import { Eye, Target, Zap, CheckCircle, AlertCircle } from 'lucide-react';

const RecognitionDisplay = ({ recognizedLabel, confidence, recognitionSuccess }) => {
    if (!recognizedLabel && !recognitionSuccess) return null;

    const confidencePercentage = confidence ? Math.round(confidence * 100) : 0;
    const isHighConfidence = confidencePercentage >= 70;
    const isMediumConfidence = confidencePercentage >= 40;

    return (
        <motion.div
            className="max-w-4xl mx-auto mb-6"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
        >
            <div className="bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-xl p-6 shadow-lg">
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-4">
                        {/* Recognition Icon */}
                        <div className={`w-12 h-12 rounded-full flex items-center justify-center ${recognitionSuccess
                                ? 'bg-gradient-to-br from-blue-500 to-indigo-600 text-white'
                                : 'bg-gray-300 text-gray-600'
                            }`}>
                            {recognitionSuccess ? <Eye size={24} /> : <AlertCircle size={24} />}
                        </div>

                        {/* Recognition Info */}
                        <div>
                            <div className="flex items-center gap-2 mb-1">
                                <h3 className="text-lg font-semibold text-gray-800">
                                    🎯 Doodle Detected
                                </h3>
                                {recognitionSuccess && (
                                    <CheckCircle size={18} className="text-green-500" />
                                )}
                            </div>

                            {recognizedLabel ? (
                                <div className="flex items-center gap-3">
                                    <span className="text-2xl font-bold text-blue-600 capitalize">
                                        {recognizedLabel.replace(/-/g, ' ')}
                                    </span>

                                    {confidence && (
                                        <div className="flex items-center gap-2">
                                            <div className="text-sm text-gray-600">
                                                Confidence:
                                            </div>
                                            <div className={`px-3 py-1 rounded-full text-sm font-medium ${isHighConfidence
                                                    ? 'bg-green-100 text-green-700 border border-green-200'
                                                    : isMediumConfidence
                                                        ? 'bg-yellow-100 text-yellow-700 border border-yellow-200'
                                                        : 'bg-red-100 text-red-700 border border-red-200'
                                                }`}>
                                                {confidencePercentage}%
                                            </div>
                                        </div>
                                    )}
                                </div>
                            ) : (
                                <div className="text-gray-600">
                                    Unable to recognize the doodle clearly
                                </div>
                            )}
                        </div>
                    </div>

                    {/* Confidence Meter */}
                    {confidence && (
                        <div className="flex flex-col items-end">
                            <div className="text-xs text-gray-500 mb-1">Recognition Strength</div>
                            <div className="w-24 h-2 bg-gray-200 rounded-full overflow-hidden">
                                <motion.div
                                    className={`h-full rounded-full ${isHighConfidence
                                            ? 'bg-gradient-to-r from-green-400 to-green-500'
                                            : isMediumConfidence
                                                ? 'bg-gradient-to-r from-yellow-400 to-yellow-500'
                                                : 'bg-gradient-to-r from-red-400 to-red-500'
                                        }`}
                                    initial={{ width: 0 }}
                                    animate={{ width: `${confidencePercentage}%` }}
                                    transition={{ duration: 1, delay: 0.3 }}
                                />
                            </div>
                        </div>
                    )}
                </div>

                {/* Additional Info */}
                <div className="mt-4 pt-4 border-t border-blue-200">
                    <div className="flex items-center justify-between text-sm">
                        <div className="flex items-center gap-2 text-blue-600">
                            <Zap size={14} />
                            <span>AI Recognition powered by MobileNet</span>
                        </div>

                        {recognitionSuccess && (
                            <div className="flex items-center gap-2 text-green-600">
                                <Target size={14} />
                                <span>Ready for style transformation</span>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </motion.div>
    );
};

export default RecognitionDisplay;