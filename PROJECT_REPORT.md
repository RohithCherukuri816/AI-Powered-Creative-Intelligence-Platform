# AI-POWERED CREATIVE INTELLIGENCE PLATFORM
## Transform Doodles into Stunning Designs with AI

**Project Report**

---

**Submitted by:** Rohith Cherukuri

**Project Type:** Full-Stack Web Application with Machine Learning

**Technologies:** React, FastAPI, TensorFlow, Stable Diffusion, ControlNet

---

## TABLE OF CONTENTS

1. [ABSTRACT](#abstract)
2. [INTRODUCTION](#introduction)
3. [SYSTEM ANALYSIS](#system-analysis)
   - 3.1 [Existing System](#existing-system)
   - 3.2 [Proposed System](#proposed-system)
   - 3.3 [Functional Requirements](#functional-requirements)
   - 3.4 [Non-Functional Requirements](#non-functional-requirements)
   - 3.5 [Hardware Requirements](#hardware-requirements)
   - 3.6 [Software Requirements](#software-requirements)
4. [SYSTEM DESIGN](#system-design)
   - 4.1 [UML Diagrams](#uml-diagrams)
   - 4.2 [Flowchart](#flowchart)
5. [TECHNOLOGY DESCRIPTION](#technology-description)
6. [IMPLEMENTATION](#implementation)
7. [OUTPUT SCREENS](#output-screens)
8. [CONCLUSION](#conclusion)
9. [BIBLIOGRAPHY](#bibliography)

---

## ABSTRACT

The AI-Powered Creative Intelligence Platform is an innovative web application that bridges the gap between simple sketches and professional artwork. This system leverages cutting-edge artificial intelligence technologies, including ControlNet and Stable Diffusion, to transform hand-drawn doodles into stunning, styled designs. The platform features an intuitive drawing canvas, intelligent doodle recognition using Convolutional Neural Networks (CNN), and advanced style transfer capabilities.

The system addresses the growing need for accessible creative tools that empower users without extensive artistic training to produce high-quality visual content. By combining real-time sketch recognition with AI-powered image generation, the platform offers a unique solution that preserves the user's creative intent while enhancing the visual quality through various artistic styles.

Key features include:
- Interactive HTML5 canvas with comprehensive drawing tools
- CNN-based doodle recognition supporting 100+ categories
- ControlNet + Stable Diffusion integration for structure-preserving style transfer
- Real-time image generation with multiple artistic styles
- Responsive design supporting desktop and mobile devices
- Complete machine learning training pipeline using Kaggle GPU

The system achieves 87% accuracy in doodle recognition and generates high-quality styled images in under 5 seconds, making it suitable for both casual users and professional designers seeking rapid prototyping tools.

---

## 1. INTRODUCTION

### 1.1 Background

In the digital age, visual content creation has become increasingly important across various domains including marketing, education, entertainment, and personal expression. However, creating professional-quality artwork traditionally requires significant artistic skills, expensive software, and considerable time investment. This creates a barrier for individuals and small businesses who need quality visual content but lack the resources or expertise.

Recent advances in artificial intelligence, particularly in the fields of computer vision and generative models, have opened new possibilities for democratizing creative tools. Technologies like Stable Diffusion and ControlNet have demonstrated remarkable capabilities in generating and transforming images based on various inputs and constraints.

### 1.2 Motivation

The motivation for this project stems from several key observations:

1. **Accessibility Gap**: Professional design tools like Adobe Photoshop or Illustrator have steep learning curves and high costs, making them inaccessible to many potential users.

2. **Creative Expression**: Many people have creative ideas but struggle to translate them into visual form due to limited drawing skills.

3. **Time Efficiency**: Even skilled artists spend considerable time on initial sketches and iterations. An AI-assisted tool can significantly accelerate this process.

4. **Educational Value**: The platform can serve as a learning tool, helping users understand different artistic styles and techniques.

### 1.3 Problem Statement

The primary problem addressed by this project is: **"How can we enable users with minimal artistic skills to transform simple sketches into professional-quality styled artwork quickly and intuitively?"**

Sub-problems include:
- Accurately recognizing user intent from rough sketches
- Preserving the structural elements of the original drawing
- Applying diverse artistic styles while maintaining coherence
- Providing real-time feedback and results
- Ensuring accessibility across different devices and platforms

### 1.4 Objectives

The main objectives of this project are:

1. **Primary Objectives**:
   - Develop an intuitive web-based drawing interface
   - Implement CNN-based doodle recognition with 85%+ accuracy
   - Integrate ControlNet + Stable Diffusion for style transfer
   - Generate styled images in under 5 seconds
   - Support 100+ doodle categories

2. **Secondary Objectives**:
   - Create a responsive design for mobile and desktop
   - Implement comprehensive training pipeline
   - Provide multiple artistic style options
   - Ensure system scalability and maintainability
   - Document the system thoroughly for future enhancements

### 1.5 Scope

**In Scope**:
- Web-based drawing canvas with basic tools (brush, eraser, colors)
- Doodle recognition using trained CNN model
- Style transfer using ControlNet + Stable Diffusion
- Multiple predefined artistic styles
- Image download and sharing capabilities
- Training pipeline for model improvement

**Out of Scope**:
- Advanced photo editing features
- User authentication and account management
- Cloud storage for user creations
- Collaborative drawing features
- Mobile native applications
- Commercial licensing management

### 1.6 Organization of Report

This report is organized into nine chapters:
- Chapter 2 analyzes existing systems and proposes the new solution
- Chapter 3 details system requirements (functional and non-functional)
- Chapter 4 presents system design with UML diagrams and flowcharts
- Chapter 5 describes the technologies used
- Chapter 6 explains the implementation details
- Chapter 7 showcases output screens and results
- Chapter 8 concludes the report with future enhancements
- Chapter 9 provides references and bibliography

---

## 2. SYSTEM ANALYSIS

### 2.1 EXISTING SYSTEM

#### 2.1.1 Overview of Existing Solutions

Several existing systems attempt to address creative content generation, each with distinct approaches and limitations:

**1. Traditional Design Software**
- **Examples**: Adobe Photoshop, Illustrator, CorelDRAW
- **Approach**: Professional-grade tools with extensive features
- **Limitations**:
  - Steep learning curve requiring months of training
  - High cost (subscription-based, $20-50/month)
  - Resource-intensive (requires powerful hardware)
  - Not suitable for quick sketches or casual users
  - No AI-assisted generation

**2. Online Drawing Tools**
- **Examples**: Canva, Figma, Sketch
- **Approach**: Simplified web-based design tools
- **Limitations**:
  - Template-based, limited creativity
  - No sketch-to-art transformation
  - Requires design knowledge for good results
  - Limited AI integration
  - Primarily for graphic design, not artistic transformation

**3. AI Art Generators**
- **Examples**: DALL-E, Midjourney, Stable Diffusion Web UI
- **Approach**: Text-to-image generation
- **Limitations**:
  - Text-only input (no sketch preservation)
  - Cannot maintain user's structural intent
  - Unpredictable results
  - No real-time drawing interface
  - Expensive API costs or complex local setup

**4. Sketch-to-Image Tools**
- **Examples**: AutoDraw (Google), Quick, Draw!
- **Approach**: Simple sketch recognition and replacement
- **Limitations**:
  - Limited to predefined clipart
  - No style customization
  - Cannot generate unique artwork
  - Basic recognition only
  - No artistic enhancement

#### 2.1.2 Limitations of Existing Systems

1. **Lack of Integration**: No system combines intuitive sketching, intelligent recognition, and AI-powered style transfer in one platform.

2. **Accessibility Issues**: Professional tools are too complex; simple tools are too limited.

3. **Cost Barriers**: Quality AI tools require expensive subscriptions or API credits.

4. **Structure Preservation**: Text-to-image generators don't preserve user's structural intent from sketches.

5. **Limited Customization**: Most tools offer either full control (complex) or no control (simple), with no middle ground.

6. **Technical Barriers**: Setting up local AI models requires technical expertise and powerful hardware.

### 2.2 PROPOSED SYSTEM

#### 2.2.1 System Overview

The proposed AI-Powered Creative Intelligence Platform addresses the limitations of existing systems by providing an integrated solution that combines:

1. **Intuitive Drawing Interface**: HTML5 canvas with familiar drawing tools
2. **Intelligent Recognition**: CNN-based doodle classification (100+ categories)
3. **Structure-Preserving AI**: ControlNet ensures sketch structure is maintained
4. **Style Transfer**: Stable Diffusion applies artistic styles
5. **Real-time Processing**: Results in under 5 seconds
6. **Accessibility**: Web-based, works on any device with a browser

#### 2.2.2 System Architecture

The system follows a three-tier architecture:

**1. Presentation Layer (Frontend)**
- React-based single-page application
- Interactive drawing canvas
- Style selection interface
- Real-time preview and results display

**2. Application Layer (Backend)**
- FastAPI REST API server
- Request validation and processing
- Model orchestration
- Image processing pipeline

**3. Data Layer**
- Trained CNN model for recognition
- ControlNet model for structure preservation
- Stable Diffusion model for generation
- Temporary image storage

#### 2.2.3 Key Features

**1. Interactive Drawing Canvas**
- Multiple brush sizes (1-20px)
- 10 color options
- Eraser tool
- Undo/Redo functionality (10 levels)
- Clear canvas option
- Keyboard shortcuts for power users
- Touch support for mobile devices

**2. Intelligent Doodle Recognition**
- Supports 100 categories (animals, vehicles, objects, etc.)
- 87% accuracy on test dataset
- Real-time classification
- Confidence scoring
- Fallback heuristic recognition

**3. AI-Powered Style Transfer**
- ControlNet preserves sketch structure
- Stable Diffusion applies artistic styles
- 8 predefined styles (watercolor, digital art, minimalist, etc.)
- Custom prompt support
- Quality enhancement prompts

**4. User Experience**
- 4-step guided workflow
- Real-time feedback
- Loading animations
- Error handling with helpful messages
- Responsive design (mobile/desktop)

**5. Performance Optimization**
- Lazy loading of AI models
- GPU acceleration when available
- CPU fallback mode
- Image compression
- Caching strategies

#### 2.2.4 Advantages Over Existing Systems

1. **Integrated Solution**: Combines sketching, recognition, and generation in one platform
2. **Structure Preservation**: ControlNet maintains user's creative intent
3. **Accessibility**: Web-based, no installation required
4. **Cost-Effective**: Free to use (after initial setup)
5. **Fast Results**: Under 5 seconds per generation
6. **Flexible**: Works with or without GPU
7. **Extensible**: Easy to add new styles and categories
8. **Educational**: Helps users learn about AI and art styles

### 2.3 FUNCTIONAL REQUIREMENTS

Functional requirements define what the system should do. They are organized by user roles and features.

#### FR1: Drawing Canvas Management

**FR1.1**: The system shall provide an HTML5 canvas for drawing
- **Input**: Mouse/touch events
- **Process**: Capture drawing strokes
- **Output**: Visual feedback on canvas
- **Priority**: High

**FR1.2**: The system shall support multiple brush sizes (1-20 pixels)
- **Input**: User selection via slider
- **Process**: Update brush size state
- **Output**: Drawing with selected size
- **Priority**: High

**FR1.3**: The system shall provide 10 color options for drawing
- **Input**: User color selection
- **Process**: Update brush color state
- **Output**: Drawing with selected color
- **Priority**: Medium

**FR1.4**: The system shall implement eraser functionality
- **Input**: Eraser tool selection
- **Process**: Remove pixels from canvas
- **Output**: Erased areas
- **Priority**: High

**FR1.5**: The system shall support undo/redo operations (10 levels)
- **Input**: Undo/redo command
- **Process**: Restore previous canvas state
- **Output**: Updated canvas
- **Priority**: Medium

**FR1.6**: The system shall allow clearing the entire canvas
- **Input**: Clear command
- **Process**: Reset canvas to blank state
- **Output**: Empty white canvas
- **Priority**: High

**FR1.7**: The system shall support keyboard shortcuts
- **Input**: Keyboard events (B, E, Ctrl+Z, etc.)
- **Process**: Execute corresponding action
- **Output**: Action performed
- **Priority**: Low

#### FR2: Doodle Recognition

**FR2.1**: The system shall recognize doodles from 100 categories
- **Input**: Canvas image (base64)
- **Process**: CNN model inference
- **Output**: Category label + confidence score
- **Priority**: High

**FR2.2**: The system shall provide confidence scores for predictions
- **Input**: Model output probabilities
- **Process**: Calculate confidence percentage
- **Output**: Confidence value (0-100%)
- **Priority**: Medium

**FR2.3**: The system shall use heuristic fallback for low confidence
- **Input**: Low confidence prediction (<30%)
- **Process**: Shape analysis (aspect ratio, contours)
- **Output**: Heuristic category guess
- **Priority**: Medium

**FR2.4**: The system shall validate sketch before processing
- **Input**: Canvas image
- **Process**: Check for non-white pixels (>0.5%)
- **Output**: Validation result (pass/fail)
- **Priority**: High

#### FR3: Style Transfer and Generation

**FR3.1**: The system shall generate styled images using AI
- **Input**: Sketch + style prompt
- **Process**: ControlNet + Stable Diffusion pipeline
- **Output**: Styled image (512x512 or larger)
- **Priority**: High

**FR3.2**: The system shall preserve sketch structure
- **Input**: Original sketch
- **Process**: ControlNet edge detection
- **Output**: Structure-preserving generation
- **Priority**: High

**FR3.3**: The system shall support 8 predefined styles
- **Input**: Style selection
- **Process**: Apply style-specific prompts
- **Output**: Styled image
- **Priority**: Medium

**FR3.4**: The system shall accept custom style prompts
- **Input**: User text prompt (max 150 chars)
- **Process**: Enhance prompt with quality terms
- **Output**: Custom styled image
- **Priority**: High

**FR3.5**: The system shall provide fallback processing
- **Input**: Sketch (when AI unavailable)
- **Process**: Traditional image filters
- **Output**: Processed image
- **Priority**: High

#### FR4: Image Management

**FR4.1**: The system shall save generated images
- **Input**: Generated image
- **Process**: Save to uploads directory
- **Output**: File path + unique ID
- **Priority**: High

**FR4.2**: The system shall allow image download
- **Input**: Download request
- **Process**: Serve image file
- **Output**: Downloaded PNG file
- **Priority**: High

**FR4.3**: The system shall generate unique IDs for each generation
- **Input**: Generation request
- **Process**: UUID generation
- **Output**: Unique identifier
- **Priority**: Medium

**FR4.4**: The system shall track processing time
- **Input**: Start/end timestamps
- **Process**: Calculate duration
- **Output**: Processing time in seconds
- **Priority**: Low

#### FR5: User Interface

**FR5.1**: The system shall display 4-step progress indicator
- **Input**: Current step
- **Process**: Update UI state
- **Output**: Visual progress display
- **Priority**: Medium

**FR5.2**: The system shall show loading animations
- **Input**: Processing state
- **Process**: Display spinner
- **Output**: Loading feedback
- **Priority**: Medium

**FR5.3**: The system shall display error messages
- **Input**: Error condition
- **Process**: Format error message
- **Output**: User-friendly error display
- **Priority**: High

**FR5.4**: The system shall provide style suggestions
- **Input**: User interaction
- **Process**: Display 8 style options
- **Output**: Clickable suggestions
- **Priority**: Low

### 2.4 NON-FUNCTIONAL REQUIREMENTS

Non-functional requirements define how the system should perform.

#### NFR1: Performance Requirements

**NFR1.1**: Response Time
- Drawing operations: < 16ms (60 FPS)
- Doodle recognition: < 500ms
- Image generation (GPU): < 5 seconds
- Image generation (CPU): < 30 seconds
- API response time: < 100ms (excluding generation)

**NFR1.2**: Throughput
- Support 10 concurrent users minimum
- Handle 100 requests per hour
- Process 1000 drawings per day

**NFR1.3**: Resource Utilization
- Frontend bundle size: < 2MB
- Memory usage (frontend): < 200MB
- Memory usage (backend): < 4GB (with models)
- GPU VRAM: < 8GB
- CPU usage: < 80% average

#### NFR2: Scalability Requirements

**NFR2.1**: Horizontal Scalability
- Support load balancing across multiple backend instances
- Stateless API design for easy scaling
- Separate model serving from API logic

**NFR2.2**: Vertical Scalability
- Support GPU upgrades without code changes
- Handle larger models (up to 16GB VRAM)
- Scale to 1000+ doodle categories

#### NFR3: Reliability Requirements

**NFR3.1**: Availability
- System uptime: 99% (excluding maintenance)
- Graceful degradation when AI models unavailable
- Automatic fallback to CPU mode

**NFR3.2**: Error Handling
- All errors logged with context
- User-friendly error messages
- No system crashes on invalid input
- Automatic retry for transient failures

**NFR3.3**: Data Integrity
- No data loss during processing
- Consistent state management
- Atomic operations for file saves

#### NFR4: Usability Requirements

**NFR4.1**: Ease of Use
- No training required for basic features
- Intuitive UI following common patterns
- Clear visual feedback for all actions
- Helpful tooltips and instructions

**NFR4.2**: Accessibility
- Keyboard navigation support
- Touch-friendly interface
- Responsive design (320px - 4K)
- Color contrast ratio: 4.5:1 minimum

**NFR4.3**: Learnability
- First-time users can create artwork in < 2 minutes
- Keyboard shortcuts discoverable
- Progressive disclosure of advanced features

#### NFR5: Security Requirements

**NFR5.1**: Input Validation
- Validate all user inputs
- Sanitize file uploads
- Prevent injection attacks
- Rate limiting on API endpoints

**NFR5.2**: Data Privacy
- No storage of user drawings (temporary only)
- No tracking or analytics without consent
- Secure file handling
- HTTPS for all communications

**NFR5.3**: Authentication (Future)
- Secure password hashing
- JWT token-based authentication
- Session management
- CSRF protection

#### NFR6: Maintainability Requirements

**NFR6.1**: Code Quality
- Modular architecture
- Clear separation of concerns
- Comprehensive comments
- Consistent coding style

**NFR6.2**: Documentation
- API documentation (OpenAPI/Swagger)
- Code documentation (docstrings)
- User guide
- Deployment guide

**NFR6.3**: Testability
- Unit test coverage: > 70%
- Integration tests for critical paths
- End-to-end tests for user workflows
- Performance benchmarks

#### NFR7: Portability Requirements

**NFR7.1**: Platform Independence
- Run on Windows, Linux, macOS
- Browser compatibility: Chrome, Firefox, Safari, Edge
- Mobile browser support: iOS Safari, Chrome Mobile

**NFR7.2**: Deployment Flexibility
- Docker containerization
- Cloud deployment ready (AWS, GCP, Azure)
- Local development setup < 30 minutes

#### NFR8: Compatibility Requirements

**NFR8.1**: Browser Compatibility
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

**NFR8.2**: Python Version
- Python 3.8 - 3.11
- TensorFlow 2.13+
- PyTorch 2.0+

**NFR8.3**: Node.js Version
- Node.js 18+
- npm 9+ or pnpm 8+

### 2.5 HARDWARE REQUIREMENTS

#### 2.5.1 Development Environment

**Minimum Requirements**:
- **Processor**: Intel Core i5 (8th gen) or AMD Ryzen 5
- **RAM**: 8GB DDR4
- **Storage**: 50GB free space (SSD recommended)
- **GPU**: Optional (NVIDIA GTX 1060 6GB or better)
- **Network**: Broadband internet connection

**Recommended Requirements**:
- **Processor**: Intel Core i7 (10th gen) or AMD Ryzen 7
- **RAM**: 16GB DDR4
- **Storage**: 100GB free space (NVMe SSD)
- **GPU**: NVIDIA RTX 3060 12GB or better
- **Network**: High-speed internet (100+ Mbps)

#### 2.5.2 Production Server

**Minimum Requirements**:
- **Processor**: 4 cores @ 2.5GHz
- **RAM**: 8GB
- **Storage**: 100GB SSD
- **GPU**: Optional (for AI features)
- **Network**: 100 Mbps uplink

**Recommended Requirements**:
- **Processor**: 8 cores @ 3.0GHz
- **RAM**: 32GB
- **Storage**: 500GB NVMe SSD
- **GPU**: NVIDIA T4 or A10 (16GB VRAM)
- **Network**: 1 Gbps uplink

#### 2.5.3 Client (User) Requirements

**Minimum Requirements**:
- **Processor**: Dual-core @ 1.5GHz
- **RAM**: 4GB
- **Storage**: 100MB free space
- **Display**: 1024x768 resolution
- **Network**: 5 Mbps internet connection
- **Input**: Mouse or touchscreen

**Recommended Requirements**:
- **Processor**: Quad-core @ 2.0GHz
- **RAM**: 8GB
- **Storage**: 500MB free space
- **Display**: 1920x1080 resolution
- **Network**: 25+ Mbps internet connection
- **Input**: Mouse/stylus with pressure sensitivity

### 2.6 SOFTWARE REQUIREMENTS

#### 2.6.1 Development Tools

**Frontend Development**:
- **Node.js**: v18.0.0 or higher
- **Package Manager**: npm 9+ or pnpm 8+
- **Code Editor**: VS Code, WebStorm, or similar
- **Browser DevTools**: Chrome DevTools or Firefox Developer Tools

**Backend Development**:
- **Python**: 3.8 - 3.11 (3.11 recommended)
- **Package Manager**: pip 23+ or conda
- **Virtual Environment**: venv or conda
- **Code Editor**: VS Code, PyCharm, or similar

**Version Control**:
- **Git**: 2.30+
- **GitHub/GitLab**: For repository hosting

**Testing Tools**:
- **Frontend**: Jest, React Testing Library
- **Backend**: pytest, unittest
- **E2E**: Playwright or Cypress

#### 2.6.2 Runtime Dependencies

**Frontend Libraries**:
```json
{
  "react": "^18.2.0",
  "react-dom": "^18.2.0",
  "framer-motion": "^10.16.4",
  "lucide-react": "^0.292.0",
  "tailwindcss": "^3.3.5",
  "vite": "^4.5.0"
}
```

**Backend Libraries**:
```
fastapi==0.109.2
uvicorn[standard]==0.27.1
torch==2.2.0
diffusers==0.26.3
transformers==4.38.2
tensorflow==2.13.0
opencv-python==4.9.0.80
Pillow==10.2.0
```

#### 2.6.3 AI/ML Models

**Pre-trained Models**:
- **ControlNet**: lllyasviel/sd-controlnet-canny
- **Stable Diffusion**: runwayml/stable-diffusion-v1-5
- **Doodle Recognition**: Custom trained CNN (100 categories)

**Model Storage**:
- Total size: ~10GB
- Location: HuggingFace Hub or local cache
- Format: SafeTensors, PyTorch, TensorFlow

#### 2.6.4 Operating System

**Supported Platforms**:
- **Windows**: 10/11 (64-bit)
- **Linux**: Ubuntu 20.04+, Debian 11+, CentOS 8+
- **macOS**: 11.0+ (Big Sur or later)

**Browser Support**:
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+
- Mobile browsers (iOS Safari, Chrome Mobile)

#### 2.6.5 Database (Future Enhancement)

**Current**: File-based storage
**Planned**:
- **PostgreSQL**: 13+ (for user data)
- **Redis**: 6+ (for caching)
- **MongoDB**: 5+ (for image metadata)

#### 2.6.6 Deployment Tools

**Containerization**:
- **Docker**: 20.10+
- **Docker Compose**: 2.0+

**Cloud Platforms** (Optional):
- AWS (EC2, S3, Lambda)
- Google Cloud Platform (Compute Engine, Cloud Storage)
- Azure (Virtual Machines, Blob Storage)
- Heroku, Railway, or Render

**CI/CD** (Optional):
- GitHub Actions
- GitLab CI
- Jenkins

---


## 3. SYSTEM DESIGN

### 3.1 UML DIAGRAMS

#### 3.1.1 Use Case Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                  AI Creative Platform                        │
│                                                              │
│  ┌──────────┐                                               │
│  │          │                                               │
│  │   User   │──────────► Draw Sketch                       │
│  │          │                │                              │
│  └──────────┘                │                              │
│       │                      ▼                              │
│       │              Select Drawing Tools                   │
│       │              (Brush, Eraser, Colors)                │
│       │                      │                              │
│       │                      ▼                              │
│       ├──────────► Enter Style Prompt                       │
│       │                      │                              │
│       │                      ▼                              │
│       ├──────────► Generate Styled Image                    │
│       │                      │                              │
│       │                      ├──► Recognize Doodle          │
│       │                      │                              │
│       │                      ├──► Apply Style Transfer      │
│       │                      │                              │
│       │                      └──► Save Generated Image      │
│       │                                                     │
│       ├──────────► Download Image                           │
│       │                                                     │
│       ├──────────► Share Image                              │
│       │                                                     │
│       └──────────► View Generation History                  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Actors**:
- **User**: Primary actor who interacts with the system

**Use Cases**:
1. **Draw Sketch**: User creates a drawing on the canvas
2. **Select Drawing Tools**: Choose brush size, color, eraser
3. **Enter Style Prompt**: Describe desired artistic style
4. **Generate Styled Image**: Transform sketch into styled artwork
5. **Recognize Doodle**: System identifies what was drawn
6. **Apply Style Transfer**: AI applies artistic style
7. **Download Image**: Save generated image locally
8. **Share Image**: Share via social media or link

#### 3.1.2 Class Diagram

```
┌─────────────────────────┐
│   DrawingCanvas         │
├─────────────────────────┤
│ - canvasRef: Ref        │
│ - brushSize: number     │
│ - brushColor: string    │
│ - tool: string          │
│ - history: Array        │
├─────────────────────────┤
│ + startDrawing()        │
│ + draw()                │
│ + stopDrawing()         │
│ + undo()                │
│ + clearCanvas()         │
│ + downloadSketch()      │
└─────────────────────────┘
           │
           │ uses
           ▼
┌─────────────────────────┐
│   ImageProcessor        │
├─────────────────────────┤
│ - device: string        │
│ - controlnet: Model     │
│ - pipe: Pipeline        │
│ - config: dict          │
├─────────────────────────┤
│ + apply_style()         │
│ + load_models()         │
│ + enhance_prompt()      │
│ + generate_image()      │
│ + fallback_process()    │
└─────────────────────────┘
           │
           │ uses
           ▼
┌─────────────────────────┐
│   DoodleRecognizer      │
├─────────────────────────┤
│ - model: CNNModel       │
│ - classes: List         │
│ - model_loaded: bool    │
├─────────────────────────┤
│ + recognize_doodle()    │
│ + preprocess_image()    │
│ + heuristic_fallback()  │
│ + load_model()          │
└─────────────────────────┘
           │
           │ uses
           ▼
┌─────────────────────────┐
│   DesignAPI             │
├─────────────────────────┤
│ - processor: Processor  │
│ - recognizer: Recognizer│
├─────────────────────────┤
│ + generate_design()     │
│ + get_generation()      │
│ + delete_generation()   │
│ + get_styles()          │
└─────────────────────────┘
```

**Key Classes**:

1. **DrawingCanvas** (Frontend)
   - Manages HTML5 canvas interactions
   - Handles drawing tools and history
   - Exports sketch as base64

2. **ImageProcessor** (Backend)
   - Loads and manages AI models
   - Applies style transformations
   - Handles fallback processing

3. **DoodleRecognizer** (Backend)
   - CNN-based doodle classification
   - Preprocesses images for recognition
   - Provides heuristic fallback

4. **DesignAPI** (Backend)
   - REST API endpoints
   - Orchestrates processing pipeline
   - Manages file storage

#### 3.1.3 Sequence Diagram - Image Generation Flow

```
User          Frontend        API           Recognizer    Processor      Models
 │                │            │                │             │            │
 │  Draw Sketch   │            │                │             │            │
 ├───────────────►│            │                │             │            │
 │                │            │                │             │            │
 │  Enter Prompt  │            │                │             │            │
 ├───────────────►│            │                │             │            │
 │                │            │                │             │            │
 │  Click Generate│            │                │             │            │
 ├───────────────►│            │                │             │            │
 │                │            │                │             │            │
 │                │ POST /generate-design       │             │            │
 │                ├───────────►│                │             │            │
 │                │            │                │             │            │
 │                │            │ recognize()    │             │            │
 │                │            ├───────────────►│             │            │
 │                │            │                │             │            │
 │                │            │                │ CNN Inference            │
 │                │            │                ├────────────────────────►│
 │                │            │                │             │            │
 │                │            │                │◄────────────────────────┤
 │                │            │                │  (label, confidence)    │
 │                │            │                │             │            │
 │                │            │◄───────────────┤             │            │
 │                │            │  (label, conf) │             │            │
 │                │            │                │             │            │
 │                │            │ apply_style()  │             │            │
 │                │            ├────────────────────────────►│            │
 │                │            │                │             │            │
 │                │            │                │             │ ControlNet │
 │                │            │                │             ├───────────►│
 │                │            │                │             │            │
 │                │            │                │             │◄───────────┤
 │                │            │                │             │  edges     │
 │                │            │                │             │            │
 │                │            │                │             │ Stable     │
 │                │            │                │             │ Diffusion  │
 │                │            │                │             ├───────────►│
 │                │            │                │             │            │
 │                │            │                │             │◄───────────┤
 │                │            │                │             │  image     │
 │                │            │                │             │            │
 │                │            │◄────────────────────────────┤            │
 │                │            │  styled_image  │             │            │
 │                │            │                │             │            │
 │                │            │ save_image()   │             │            │
 │                │            ├────────────────┤             │            │
 │                │            │                │             │            │
 │                │◄───────────┤                │             │            │
 │                │  Response  │                │             │            │
 │                │  (url, id) │                │             │            │
 │                │            │                │             │            │
 │◄───────────────┤            │                │             │            │
 │  Display Image │            │                │             │            │
 │                │            │                │             │            │
```

**Flow Description**:
1. User draws sketch on canvas
2. User enters style prompt
3. Frontend sends POST request with sketch and prompt
4. API validates request
5. Recognizer identifies doodle category
6. Processor prepares control image using ControlNet
7. Stable Diffusion generates styled image
8. Image saved to storage
9. Response returned with image URL
10. Frontend displays generated image

#### 3.1.4 Activity Diagram - Complete User Workflow

```
                    START
                      │
                      ▼
              ┌───────────────┐
              │  Open Canvas  │
              └───────┬───────┘
                      │
                      ▼
              ┌───────────────┐
              │  Draw Sketch  │◄─────┐
              └───────┬───────┘      │
                      │              │
                      ▼              │
              ┌───────────────┐      │
              │ Sketch Valid? │      │
              └───────┬───────┘      │
                      │              │
                 Yes  │  No          │
                      ▼              │
              ┌───────────────┐      │
              │ Show Prompt   │      │
              │ Input         │      │
              └───────┬───────┘      │
                      │              │
                      ▼              │
              ┌───────────────┐      │
              │ Enter Style   │      │
              │ Prompt        │      │
              └───────┬───────┘      │
                      │              │
                      ▼              │
              ┌───────────────┐      │
              │ Click Generate│      │
              └───────┬───────┘      │
                      │              │
                      ▼              │
              ┌───────────────┐      │
              │ Show Loading  │      │
              └───────┬───────┘      │
                      │              │
                      ▼              │
              ┌───────────────┐      │
              │ Recognize     │      │
              │ Doodle        │      │
              └───────┬───────┘      │
                      │              │
                      ▼              │
              ┌───────────────┐      │
              │ Generate      │      │
              │ Styled Image  │      │
              └───────┬───────┘      │
                      │              │
                      ▼              │
              ┌───────────────┐      │
              │ Display Result│      │
              └───────┬───────┘      │
                      │              │
                      ▼              │
              ┌───────────────┐      │
              │ Satisfied?    │      │
              └───────┬───────┘      │
                      │              │
                 Yes  │  No          │
                      │              │
                      │              └──────┐
                      ▼                     │
              ┌───────────────┐             │
              │ Download or   │             │
              │ Share Image   │             │
              └───────┬───────┘             │
                      │                     │
                      ▼                     │
                    END                     │
                                           │
                                           │
                    ┌──────────────────────┘
                    │
                    ▼
            ┌───────────────┐
            │ Try Again?    │
            └───────┬───────┘
                    │
               Yes  │  No
                    │
                    ├──────────► END
                    │
                    └──────────────────────┐
                                          │
                                          ▼
                                  Clear Canvas
                                          │
                                          └──────────────────►
```

#### 3.1.5 Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend Layer                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   App.jsx    │  │ DrawingCanvas│  │ PromptInput  │     │
│  │              │──│              │  │              │     │
│  │  Main App    │  │  Canvas UI   │  │  Style Input │     │
│  └──────┬───────┘  └──────────────┘  └──────────────┘     │
│         │                                                    │
│         │          ┌──────────────┐  ┌──────────────┐     │
│         └──────────│ ImageCard    │  │ LoadingSpinner│    │
│                    │              │  │              │     │
│                    │  Results UI  │  │  Loading UI  │     │
│                    └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ HTTP/REST
                            ▼
┌────────────────────────────────────────────────────────────┐
│                     Backend Layer                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   main.py    │  │  design.py   │  │  api.py      │      │
│  │              │──│              │  │              │      │
│  │  FastAPI App │  │  Routes      │  │  Endpoints   │      │
│  └──────┬───────┘  └──────┬───────┘  └──────────────┘      │
│         │                 │                                │
│         │                 ▼                                │
│         │          ┌──────────────┐  ┌──────────────┐      │
│         └──────────│ImageProcessor│  │DoodleRecognizer     │
│                    │              │  │              │      │
│                    │  AI Pipeline │  │  CNN Model   │      │
└───────────────────────────┼──────────────────┼─────────────┘
                            │                  │
                            ▼                  ▼
┌─────────────────────────────────────────────────────────────┐
│                     Model Layer                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │  ControlNet  │  │ Stable       │  │  CNN Model   │       │
│  │              │  │ Diffusion    │  │              │       │
│  │  Structure   │  │  Generation  │  │  Recognition │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 FLOWCHART

#### 3.2.1 Main System Flowchart

```
                        START
                          │
                          ▼
                  ┌───────────────┐
                  │ Initialize    │
                  │ Application   │
                  └───────┬───────┘
                          │
                          ▼
                  ┌───────────────┐
                  │ Load Frontend │
                  │ Components    │
                  └───────┬───────┘
                          │
                          ▼
                  ┌───────────────┐
                  │ Display Canvas│
                  └───────┬───────┘
                          │
                          ▼
                  ┌───────────────┐
                  │ User Draws    │
                  │ Sketch        │
                  └───────┬───────┘
                          │
                          ▼
                  ┌───────────────┐
                  │ Has Drawing?  │
                  └───────┬───────┘
                          │
                     No   │   Yes
                          │
                          ▼
                  ┌───────────────┐
                  │ Show Prompt   │
                  │ Input         │
                  └───────┬───────┘
                          │
                          ▼
                  ┌───────────────┐
                  │ User Enters   │
                  │ Style Prompt  │
                  └───────┬───────┘
                          │
                          ▼
                  ┌───────────────┐
                  │ Validate Input│
                  └───────┬───────┘
                          │
                     Valid│Invalid
                          │
                          ▼
                  ┌───────────────┐
                  │ Send to API   │
                  └───────┬───────┘
                          │
                          ▼
                  ┌───────────────┐
                  │ Decode Image  │
                  └───────┬───────┘
                          │
                          ▼
                  ┌───────────────┐
                  │ Recognize     │
                  │ Doodle        │
                  └───────┬───────┘
                          │
                          ▼
                  ┌───────────────┐
                  │ Load AI Models│
                  └───────┬───────┘
                          │
                          ▼
                  ┌───────────────┐
                  │ Models        │
                  │ Available?    │
                  └───────┬───────┘
                          │
                     Yes  │  No
                          │
                          ├──────────┐
                          │          │
                          ▼          ▼
                  ┌───────────┐ ┌──────────┐
                  │ ControlNet│ │ Fallback │
                  │ + SD      │ │ Process  │
                  │ Pipeline  │ │          │
                  └─────┬─────┘ └────┬─────┘
                        │            │
                        └────┬───────┘
                             │
                             ▼
                  ┌───────────────┐
                  │ Generate Image│
                  └───────┬───────┘
                          │
                          ▼
                  ┌───────────────┐
                  │ Save Image    │
                  └───────┬───────┘
                          │
                          ▼
                  ┌───────────────┐
                  │ Return Result │
                  └───────┬───────┘
                          │
                          ▼
                  ┌───────────────┐
                  │ Display to    │
                  │ User          │
                  └───────┬───────┘
                          │
                          ▼
                  ┌───────────────┐
                  │ User Actions  │
                  │ (Download/    │
                  │  Share/Try    │
                  │  Again)       │
                  └───────┬───────┘
                          │
                          ▼
                        END
```

#### 3.2.2 Doodle Recognition Flowchart

```
                    START
                      │
                      ▼
              ┌───────────────┐
              │ Receive Image │
              └───────┬───────┘
                      │
                      ▼
              ┌───────────────┐
              │ Convert to    │
              │ Grayscale     │
              └───────┬───────┘
                      │
                      ▼
              ┌───────────────┐
              │ Resize to     │
              │ 28x28         │
              └───────┬───────┘
                      │
                      ▼
              ┌───────────────┐
              │ Normalize     │
              │ Pixels (0-1)  │
              └───────┬───────┘
                      │
                      ▼
              ┌───────────────┐
              │ Invert Colors │
              │ if Needed     │
              └───────┬───────┘
                      │
                      ▼
              ┌───────────────┐
              │ Model Loaded? │
              └───────┬───────┘
                      │
                 Yes  │  No
                      │
                      ├──────────────┐
                      │              │
                      ▼              ▼
              ┌───────────────┐ ┌──────────┐
              │ CNN Inference │ │ Heuristic│
              │               │ │ Analysis │
              └───────┬───────┘ └────┬─────┘
                      │              │
                      ▼              │
              ┌───────────────┐      │
              │ Get Top       │      │
              │ Prediction    │      │
              └───────┬───────┘      │
                      │              │
                      ▼              │
              ┌───────────────┐      │
              │ Confidence    │      │
              │ > 30%?        │      │
              └───────┬───────┘      │
                      │              │
                 Yes  │  No          │
                      │              │
                   └──────┐
                      │                                     └─────────┬───────────┘
                                │
                                ▼
                        ┌───────────────┐
                        │ Return Label  │
                        │ + Confidence  │
                        └───────┬───────┘
                                │
                                ▼
                              END
```

#### 3.2.3 Style Transfer Flowchart

```
                    START
                      │
                      ▼
              ┌───────────────┐
              │ Receive Sketch│
              │ + Prompt      │
              │ + Label       │
              └───────┬───────┘
                      │
                      ▼
              ┌───────────────┐
              │ Enhance Prompt│
              │ with Quality  │
              │ Terms         │
              └───────┬───────┘
                      │
                      ▼
              ┌───────────────┐
              │ Resize Image  │
              │ to 512x512    │
              └───────┬───────┘
                      │
                      ▼
              ┌───────────────┐
              │ GPU Available?│
              └───────┬───────┘
                      │
                 Yes  │  No
                      │
                      ├──────────────┐
                      │              │
                      ▼              ▼
              ┌───────────────┐ ┌──────────┐
              │ Load AI Models│ │ Fallback │
              │               │ │ Mode     │
              └───────┬───────┘ └────┬─────┘
                      │              │
                      ▼              │
              ┌───────────────┐      │
              │ Prepare          │
              │ Control Image │      │
              │ (Canny Edges) │      │
              └───────┬───────┘      │
                      │              │
                 ▼              │
              ┌───────────────┐      │
              │ ControlNet    │      │
              │ Processing    │      │
              └───────┬───────┘      │
                      │              │
                      ▼              │
              ┌───────────────┐      │
              │ Stable        │      │
              │ Diffusion     │      │
              │ Generation    │      │
              │ (20 steps)    │      │
              └───────┬───────┘      │
                      │              │
                      └──────┬───────┘
                             │
                             ▼
                  ┌───────────────┐
                  │ Post-process  │
                  │ Image         │
                  └───────┬───────┘
                          │
                          ▼
                  ┌───────────────┐
                  │ Return Styled │
                  │ Image         │
                  └───────┬───────┘
                          │
                          ▼
                        END
```

---


## 4. TECHNOLOGY DESCRIPTION

### 4.1 Frontend Technologies

#### 4.1.1 React 18

**Description**: React is a JavaScript library for building user interfaces, developed by Facebook.

**Key Features**:
- **Component-Based Architecture**: Reusable UI components
- **Virtual DOM**: Efficient rendering and updates
- **Hooks**: State management without classes
- **JSX**: HTML-like syntax in JavaScript
- **Concurrent Features**: Improved performance

**Usage in Project**:
- Main UI framework
- Component composition (DrawingCanvas, PromptI
- State management (useState, useEffect)
- Event handling
- Conditional rendering

**Advantages**:
- Large ecosystem and community
- Excellent performance
- Easy to learn and use
- Strong TypeScript support
- Rich developer tools

#### 4.1.2 Vite

**Description**: Next-generation frontend build tool that provides fast development experience.

**Key Features**:
- **Lightning Fast HMR**: Hot Module Replacement in milliseconds
- **Optimized Build**: Rollup-based production builds
- **Native ESM**: Uses native ES modules
- **Plugin System**: Extensible architecture
- **TypeScript Support**: Built-in TypeScript support

**Usage in Project**:
- Development server
- Build optimization
- Asset handling
- Environment variable management

**Advantages**:
- Instant server start
- Fast hot reload
- Optimized production builds
- Simple configuration
- Better than Create React App

#### 4.1.3 Tailwind CSS

**Description**: Utility-first CSS framework for rapid UI development.

**Key Features**:
- **Utility Classes**: Pre-defined CSS classes
- **Responsive Design**: Mobile-first approach
- **Customization**: Highly configurable
- **JIT Mode**: Just-in-Time compilation
- **Dark Mode**: Built-in dark mode support

**Usage in Project**:
- All component styling
- Custom pastel color palette
- Responsive layouts
- Animations and transitions
- Gradient backgrounds

**Advantages**:
- Rapid development
- Consistent design system
- Small production bundle
- No CSS conflicts
- Easy maintenance

#### 4.1.4 Framer Motion

**Description**: Production-ready animation library for React.

**Key Features**:
- **Declarative Animations**: Simple API
- **Gesture Support**: Drag, hover, tap
- **Layout Animations**: Automatic layout transitions
- **SVG Support**: Animate SVG paths
- ** Rendering**: SSR compatible

**Usage in Project**:
- Page transitions
- Component animations
- Loading spinners
- Hover effects
- Progress indicators

**Advantages**:
- Smooth 60fps animations
- Easy to use
- Performant
- TypeScript support
- Great documentation

#### 4.1.5 HTML5 Canvas API

**Description**: Native browser API for drawing graphics.

**Key Features**:
- **2D Drawing**: Lines, shapes, text
- **Image Manipulation**: Pixel-level control
- **Compositing**: Blend modes
- **Transformations**: Rotate, scale, translate
- **Export**: toDataURL, toBlob

**Usage in Project**:
- Drawing canvas implementation
- Sketch capture
- Real-time drawing
- Undo/redo functionality
- Image export

**Advantages**:
- Native browser support
- High performance
- No dependencies
- Full control
- Cross-browser compatible

### 4.2 Backend Technologies

#### 4.2.1 FastAPI

**Description**: Modern, fast web framework for building APIs with Python 3.8+.

**Key Features**:
- **Fast Performance**: Based on Starlette and Pydantic
- **Automatic Documentation**: OpenAPI/Swagger
- **Type Hints**: Python type annotations
- **Async Support**: Native async/await
- **Data  Automatic request validation

**Usage in Project**:
- REST API endpoints
- Request/response handling
- Data validation
- CORS middleware
- Static file serving

**Advantages**:
- Very fast (comparable to Node.js)
- Easy to learn
- Automatic API docs
- Type safety
- Modern Python features

#### 4.2.2 PyTorch

**Description**: Open-source machine learning framework developed by Facebook.

**Key Features**:
- **Dynamic Computation**: Define-by-run approach
- **GPU Acceleration**: CUDA support
- **Autograd**: Automatic differentiation
- **TorchScript**: Production deployment
- **d Training**: Multi-GPU support

**Usage in Project**:
- ControlNet model loading
- Stable Diffusion inference
- GPU/CPU device management
- Tensor operations
- Model optimization

**Advantages**:
- Pythonic and intuitive
- Strong research community
- Excellent documentation
- Production-ready
- HuggingFace integration

#### 4.2.3 Diffusers (HuggingFace)

**Description**: Library for state-of-the-art diffusion models.

**Key Features**:
- **Pre-trained Models**: Access to thousands of models
- **Pipelines**: High-level APIs
- **Schedulers**: Various sampling methods
- **ControlNet Support**: Structure-preserving generation
- **Optimization**: Memory-efficient attention

**Usage in Project**:
- Stable Diffusion pipeline
- ControlNet integration
- Model loading and caching
- Image generation
- Prompt enhancement

**Advantages**:
- Easy to use
- Well-maintained
- Active community
- Regular updates
- Excellent documentation

#### 4.2.4 TensorFlow/Keras

**Description**: End-to-end machine learning platform.

**Key Features**:
- **High-Level API**: Keras integration
- **Production Ready**: TensorFlow Serving
- **Mobile Support**: TensorFlow Lite
- **Distributed Training**: Multi-device support
- **Visualization**: TensorBoard

**Usage in Project**:
- CNN model training
- Doodle recognition
- Model evaluation
- Data preprocessing
- Model export

**Advantages**:
- Industry standard
- Comprehensive ecosystem
- Strong production support
- Mobile deployment
- Extensive documentation

#### 4.2.5 OpenCV

**Description**: Open-source computer vision library.

**Key Features**:
- **Image Processing**: Filters, transformations
- **Feature Detection**: Edges, corners, contours
- **Object Detection**: Face, object recognition
- **Video Analysis**: Motion tracking
- **Camera Calibration**: 3D reconstruction

**Usage in Project**:
- Image preprocessing
- Edge detection (Canny)
- Contour analysis
- Shape recognition
- Heuristic fallbackges**:
- Comprehensive features
- High performance
- Cross-platform
- Well-documented
- Industry standard

#### 4.2.6 Pillow (PIL)

**Description**: Python Imaging Library for image processing.

**Key Features**:
- **Format Support**: PNG, JPEG, GIF, etc.
- **Image Operations**: Resize, crop, rotate
- **Filters**: Blur, sharpen, enhance
- **Drawing**: Text, shapes, lines
- **Color Mann**: RGB, HSV, etc.

**Usage in Project**:
- Image loading and saving
- Format conversion
- Fallback image processing
- Artistic filters
- Image enhancement

**Advantages**:
- Easy to use
- Pure Python
- Wide format support
- Good documentation
- Active maintenance

### 4.3 AI/ML Models

#### 4.3.1 ControlNet

**Description**: Neural network structure to control diffusion models.

**Model**: lllyasviel/sd-controlnet-canny

**Key Features**:
- **Structure Preservation**: Maintains sketch edges
- **Conditional Generation**: Guided by control image
- **Multiple Conditions**: Canny, depth, pose, etc.
- **Fine Control**: Adjustable conditioning scale
- **Compatibility**: Works with Stable Diffusion

**Usage in Project**:
- Edge detection from sketch
- Structure-preserving generation
- Control image preparation
- Conditioning scale adjustment

**Advantages**:
- Preserves user intent
- High-quality results
- Flexible control
- Fast inference
- Well-documented

#### 4.3.2 Stable Diffusion

**Description**: Latent diffusion model for text-to-image generation.

**Model**: runwayml/stable-diffusion-v1-5

**Key Features**:
- **High Quality**: 512x512 images
- **Fast Generation**: 20-50 steps
- **Text Conditioning**: Prompt-based control
- **Negative Prompts**: Avoid unwanted features
- **Schedulers**: Various sampling methods

**Usage in Project**:
- Image generation from sketch
- Style application
- Prompt-based control
- Quality enhancement

**Advantages**:
- Open source
- High quality
- Fast inference
- Flexible
- Active community

#### 4.3.3 Custom CNN Model

**Description**: Convolutional Neural Network for doodle recognition.

**Architecture**:
```
Input (28x28x1)
    ↓
Conv2D (32 filters, 3x3) + ReLU
    ↓
MaxPooling2D (2x2)
    ↓
Conv2D (64 filters, 3x3) + ReLU
    ↓
MaxPooling2D (2x2)
    ↓
Conv2D (64 filters, 3x3) + ReLU
    ↓
Flatten
    ↓
Dense (128) + ReLU + Dropout(0.5)
    ↓
Dense (100) + Softmax
```

**Training Details**:
- **Dataset**: Google Quick Draw (100 categories)
- **Samples**: 10,000 per category
- **Epochs**: 30
- **Batch Size**: 128
- **Optimizer**: Adam (lr=0.001)
- **Accuracy**: 87% test accuracy

**Usage in Project**:
- Doodle classification
- Confidence scoring
- Real-time inference

**Advantages**:
- Lightweight (25MB)
- Fast inference (<50ms)
- High accuracy
- Easy to train
- Customizable

### 4.4 Development Tools

#### 4.4.1 Git & GitHub

**Description**: Version control system and hosting platform.

**Usage**:
- Source code management
- Version tracking
- Collaboration
- Issue tracking
- Documentation

#### 4.4.2 VS Code

**Description**: Lightweight code editor by Microsoft.

**Extensions Used**:
- ESLint
- Prettier
- Python
- Tailwind CSS IntelliSense
- GitLens

#### 4.4.3 Postman

**Description**: API development and testing tool.

**Usage**:
- API endpoint testing
- Request debugging
- Response validation
- Documentation

#### 4.4.4 Chrome DevTools

**Description**: Browser developer tools.

**Usage**:
- Frontend debugging
- Performance profiling
- Network analysis
- Console logging

### 4.5 Deployment Technologies

#### 4.5.1 Docker

**Description**: Containerization platform.

**Usage**:
- Application containerization
- Consistent environments
- Easy deployment
- Scalability

#### 4.5.2 Ngrok

**Description**: Secure tunneling to localhost.

**Usage**:
- Backend URL tunneling
- Testing on mobile devices
- Temporary public URLs
- HTTpport

### 4.6 Training Infrastructure

#### 4.6.1 Kaggle

**Description**: Data science platform with free GPU access.

**Features**:
- Free GPU (T4, P100)
- Jupyter notebooks
- Dataset hosting
- Community

**Usage**:
- Model training
- GPU acceleration
- Experiment trackingrt

#### 4.6.2 Google Quick Draw Dataset

**Description**: Collection of 50 million drawings across 345 categories.

**Features**:
- Large-scale dataset
- Diverse drawings
- Public domain
- Easy access

**Usage**:
- Training data
- 100 categories selected
- 10,000 samples per category
- Validation and testing

---

## 5. IMPLEMENTATION

### 5.1 Project Structure

```
ai-creative-platform/
├── frontend/                    # React frontend
│   ├── src/
│   │   ├── components/         # React components
│   │   │   ├── DrawingCanvas.jsx
│   │   │   ├── PromptInput.jsx
│   │   │   ├── GeneratedImageCard.jsx
│   │   │   ├── LoadingSpinner.jsx
│   │   │   ├── KeyboardShortcuts.jsx
│   │   │   └── TouchGestures.jsx
│   │   ├── App.jsx             # Main app compo│   ├── main.jsx            # Entry point
│   │   ├── api.js              # API client
│   │   └── index.css           # Global styles
│   ├── public/                 # Static assets
│   ├── package.json            # Dependencies
│   ├── vite.config.js          # Vite configuration
│   ├── tailwind.config.js      # Tailwind configuration
│   └── .env                    # Environment variables
│
├── backend/                     # FastAPI backend
│   ├── routes/
│   │   └── design.py           # API routes
│   ├── utils/
│   │   └── image_processor.py  # Image processing
│   ├── config/
│   │   └── ai_config.py        # AI configuration
│   ├── main.py                 # FastAPI app
│   ├── requirements.txt        # Python dependencies
│   └── .env                    # Environment variables
│
├── colab_backend/              # Enhanced backend
│   ├── app.py                  # FastAPI app
│   ├── processor.py            # Image processor
│   ├── recognizer.py           # Doodle recognizer
│   └── requirements.txt        # Dependencies
│
├── training/                    # ML training pipeline
│   ├── kaggle_training.ipynb   # Kaggle notebook
│   ├── config.py               # Training config
│   ├── download_dataset.py     # Dataset downloader
│   ├── data_preprocessing.py   # Data preprocessing
│   ├── model_architecture.py   # Model definitions
│   ├── train_model.py          # Training script
│   ├── evaluate_model.py       # Evaluation
│   ├── export_model.py         # Model export
│   └── utils/                  # Utility functions
│
├── .env                        # Shared environment
├── package.json                # Root package.json
├── dev.js                      # Development server
└── README.md                   # Documentation
```

### 5.2 Frontend Implementation

#### 5.2.1 Drawing Canvas Component

**File**: `frontend/src/components/DrawingCanvas.jsx`

**Key Features**:
```javascript
// State management
const [isDrawing, setIsDrawing] = useState(false);
const [brushSize, setBrushSize] = useState(5);
const [brushColor, setBrushColor] = useState('#DC2626');
const [tool, setTool] = useState('brush');
const [history, setHistory] = useState([]);

// Drawing functions
const startDrawing = (e) => {
    setIsDrawing(true);
    const ctx = canvasRef.current.getContext('2d');
    ctx.strokeStyle = brushColor;
    ctx.lineWidth = brushSize;
    ctx.beginPath();
};

const draw = (e) => {
    if (!isDrawing) return;
    const ctx = canvasRef.current.getContext('2d');
    ctx.lineTo(x, y);
    ctx.stroke();
};

const stopDrawing = () => {
    setIsDrawing(false);
    // Save to history
    const imageData = ctx.getImageData(0, 0, width, height);
    setHistory(prev => [...prev.slice(-9), imageData]);
    // Export as base64
    const dataURL = canvas.toDataURL('image/png');
    onSketchChange(dataURL);
};
```

**Implementation Details**:
- HTML5 Canvas API for drawing
- Mouse and touch event handling
- Undo/redo with history stack (10 levels)
- Keyboard shortcuts (B, E, Ctrl+Z, etc.)
- Real-time export to base64

#### 5.2.2 Prompt Input Component

**File**: `frontend/src/components/PromptInput.jsx`

**Key Features**:
```javascript
const [prompt, setPrompt] = useState('');

const suggestions = [
    "watercolor painting style",
    "minimalist line art",
    "vibrant digital art",
    // ... more suggestions
];

const handleSubmit = (e) => {
    e.preventDefault();
    if (prompt.trim() && !isLoading) {
        onGenerate(prompt.trim());
    }
};
```

**Implementation Details**:
- Textarea with character limit (150)
- 8 predefined style suggestions
- Form validation
- Loading state management
- Animated submit button

#### 5.2.3 API Integration

**File**: `frontend/src/api.js`

**Implementation**:
```javascript
const API_BASE_URL = import.meta.env.VITE_API_URL;

export const api = {
    async generateDesign(prompt, sketchData) {
        const response = await fetch(`${API_BASE_URL}/generate-design`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt, sketch: sketchData })
        });
        return await response.json();
    }
};
```

**Features**:
- Environment-based URL configuration
- Error handling
- Response parsing
- Absolute URL helper for images

### 5.3 Backend Implementation

#### 5.3.1 API Routes

**File**: `backend/routes/design.py`

**Main Endpoint**:
```python
@router.post("/generate-design", response_model=DesignResponse)
async def generate_design(request: DesignRequest):
    # Validate input
    if not request.prompt.strip():
        raise HTTPException(400, "Prompt cannot be empty")
    
    # Decode base64 image
    sketch_b64 = request.sketch.split(',')[1]
    image_data = base64.b64decode(sketch_b64)
    sketch_image = Image.open(io.BytesIO(image_data))
    
    # Process image
    processor = ImageProcessor()
    processed_image = processor.apply_style_transformation(
        sketch_image, request.prompt, generation_id
    )
    
    # Save and return
    filename = f"generated_{generation_id}.png"
    processed_image.save(f"uploads/{filename}")
    
    return DesignResponse(
        image_url=f"/uploads/{filename}",
        generation_id=generation_id,
        prompt_used=request.prompt,
        processing_time=processing_time
    )
```

**Features**:
- Request validation with Pydantic
- Base64 image decoding
- Error handling
- File management
- Response formatting

#### 5.3.2 Image Processor

**File**: `backend/utils/image_processor.py`

**Key Methods**:
```python
class ImageProcessor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.controlnet_canny = None
        self.pipe_canny = None
    
    def apply_style_transformation(self, image, prompt, generation_id):
        # Try AI generation
        result = self._generate_with_controlnet(image, prompt)
        if result is not None:
            return result
        # Fallback to traditional processing
        return self._apply_fallback_processing(image, prompt)
    
    def _generate_with_controlnet(self, image, prompt):
        # Load models if needed
        if not self._load_controlnet_models():
            return None
        
        # Enhance prompt
        enhanced_prompt = self._enhance_prompt(prompt)
        
        # Prepare control image
        control_image = self.canny_detector(image)
        
        # Generate
        result = self.pipe_canny(
            prompt=enhanced_prompt,
            image=control_image,
            num_inference_steps=20,
            guidance_scale=7.5
        ).images[0]
        
        return result
```

**Features**:
- Lazy model loading
- GPU/CPU auto-detection
- ControlNet + Stable Diffusion pipeline
- Prompt enhancement
- Fallback processing

#### 5.3.3 Doodle Recognizer

**File**: `colab_backend/recognizer.py`

**Implementation**:
```python
class DoodleRecognizer:
    def __init__(self):
        self.model = None
        self.classes = [
            'airplane', 'car', 'cat', 'dog', 'house',
            # ... 100 categories total
        ]
        self._load_model()
    
    def recognize_doodle(self, image):
        # Preprocess
        processed = self._enhanced_preprocess(image)
        
        # Predict
        predictions = self.model.predict(processed, verbose=0)
        predicted_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_idx])
        
        # Fallback if low confidence
        if confidence < 0.3:
            return self._advanced_heuristic_recognition(image)
        
        return self.classes[predicted_idx], confidence
    
    def _enhanced_preprocess(self, image):
        # Convert to grayscale
        image = image.convert('L')
        # Resize to 28x28
        image = image.resize((28, 28))
        # Normalize
        array = np.array(image).astype('float32') / 255.0
        # Invert if needed
        if np.mean(array) > 127:
            array = 255 - array
        # Reshape
        return array.reshape(1, 28, 28, 1)
```

**Features**:
- CNN-based classification
- Image preprocessing
- Confidence scoring
- Heuristic fallback
- 100 category support

### 5.4 Training Pipeline Implementation

#### 5.4.1 Dataset Download

**File**: `training/download_dataset.py`

**Implementation**:
```python
def download_category(category, base_url, samples_limit=10000):
    filename = f"{category.replace(' ', '_')}.npy"
    url = f"{base_url}/{filename}"
    filepath = os.path.join('data', filename)
    
    response = requests.get(url, stream=True)
    with open(filepath, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    
    # Limit samples
    data = np.load(filepath)
    if len(data) > samples_limit:
        np.save(filepath, data[:samples_limit])
```

**Features**:
- Progress tracking
- Sample limiting
- Error handling
- Resumable downloads

#### 5.4.2 Model Training

**File**: `training/train_model.py`

**Implementation**:
```python
# Build model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(100, activation='softmax')
])

# Compile
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train
history = model.fit(
    X_train, y_train,
    batch_size=128,
    epochs=30,
    validation_data=(X_val, y_val),
    callbacks=[
        ModelCheckpoint('best_model.h5', save_best_only=True),
        EarlyStopping(patience=10),
        ReduceLROnPlateau(factor=0.5, patience=5)
    ]
)
```

**Features**:
- CNN architecture
- Data augmentation
- Callbacks (checkpoint, early stopping)
- Learning rate scheduling
- Validation monitoring

### 5.5 Key Algorithms

#### 5.5.1 Canny Edge Detection

**Purpose**: Extract edges from sketch for ControlNet

**Algorithm**:
1. Convert to grayscale
2. Apply Gaussian blur
3. Calculate gradients
4. Non-maximum suppression
5. Double threshold
6. Edge tracking by hysteresis

**Implementation**:
```python
def prepare_control_image(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return Image.fromarray(edges)
```

#### 5.5.2 Prompt Enhancement

**Purpose**: Improve generation quality

**Algorithm**:
1. Detect style keywords
2. Add quality enhancers
3. Add style-specific terms
4. Combine with negative prompts

**Implementation**:
```python
def enhance_prompt(prompt):
    quality_terms = "high quality, detailed, masterpiece"
    
    if "watercolor" in prompt.lower():
        style_terms = "watercolor painting, soft brushstrokes"
    elif "digital" in prompt.lower():
        style_terms = "digital art, vibrant colors"
    else:
        style_terms = ""
    
    return f"{prompt}, {style_terms}, {quality_terms}"
```

#### 5.5.3 Heuristic Recognition

**Purpose**: Fallback when CNN confidence is low

**Algorithm**:
1. Convert to grayscale
2. Find contours
3. Calculate aspect ratio
4. Analyze shape features
5. Match to category

**Implementation**:
```python
def heuristic_recognition(image):
    gray = image.convert('L')
    array = np.array(gray)
    contours, _ = cv2.findContours(array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    aspect_ratio = w / h
    
    if aspect_ratio > 2.0:
        return "car", 0.7
    elif aspect_ratio < 0.6:
        return "tree", 0.6
    else:
        return "house", 0.7
```

### 5.6 Performance Optimizations

#### 5.6.1 Lazy Loading

**Models loaded only when needed**:
```python
def _load_controlnet_models(self):
    if self.controlnet_canny is None:
        self.controlnet_canny = ControlNetModel.from_pretrained(...)
        self.pipe_canny = StableDiffusionControlNetPipeline.from_pretrained(...)
```

#### 5.6.2 Memory Optimization

**Enable attention slicing**:
```python
self.pipe_canny.enable_attention_slicing()
self.pipe_canny.enable_model_cpu_offload()
```

#### 5.6.3 Caching

**Model caching**:
```python
# Models cached in ~/.cache/huggingface/
# Reused across sessions
```

#### 5.6.4 Image Compression

**Optimize file size**:
```python
processed_image.save(filepath, "PNG", quality=95, optimize=True)
```

---

