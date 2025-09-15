# 🎉 Final Changelog v0.5.0

## 🐛 Bug Fixes
- Fixed checkpoint filename epoch numbering (now 1-based, consistent with UI)
- Fixed resume training duplicate epoch issue
- Fixed "Change (from prev)" display when resuming training
- Fixed cross-session checkpoint cleanup for best model tracking

## ✨ New Features
- Comprehensive Training Reports - Rich-formatted detailed reports with training stats, model info, and performance metrics
- Hardware Monitoring - Real-time CPU, RAM, and GPU usage tracking during training
- Enhanced Resume Training - Automatic epoch calculation, project consistency checks, simplified CLI
- Trainer Report Property - trainer.report for programmatic access to detailed training results

## 🔄 Improvements
- Migrated examples from argparse to click for consistent CLI
- Simplified resume training workflow with automatic configuration
- Enhanced training completion experience with detailed reports
- Added comprehensive documentation for resume training

## 📦 Dependencies
- Added psutil>=5.8.0 for hardware monitoring
- Added pynvml>=8.0.4 for GPU monitoring

## 📄 New Files
- treadmill/report.py - Training report system
- examples/basic_training_resume.py - Resume training example
- RESUME_TRAINING.md - Resume training documentation

## 📚 Documentation Updates
- Updated README.md with comprehensive training reports and hardware monitoring features
- Added hardware monitoring installation instructions and dependencies
- Fixed quick start examples to match actual API implementation
- Updated TrainingConfig documentation with new checkpointing options
- Enhanced resume training documentation with automatic epoch calculation
- Updated output examples to show new comprehensive training report format
- Fixed examples section to reflect actual files and CLI usage
- Corrected inconsistencies between README examples and actual implementation 