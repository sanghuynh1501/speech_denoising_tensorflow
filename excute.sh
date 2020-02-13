#!/bin/bash
cp weights.json ~/go/src/json2binary_voice
cd ~/go/src/json2binary_voice
go run main.go
cp weights.gob ~/go/src/speech_denoise_go
cp weights.gob ~/kitchen/voice_denoising_react/public
cd ~/go/src/speech_denoise_go
#go run main.go
