#!/usr/bin/env python3
"""
QuizMaster CLI Entry Point - Ragas-Inspired Question Generation System

This is the main entry point for the QuizMaster CLI application.
"""

import asyncio

if __name__ == "__main__":
    from quizmaster.cli import main
    asyncio.run(main())
