#!/bin/bash
cp weights.json ~/go/src/json2binary
cd ~/go/src/json2binary
go run main.go
cp integerdata.gob ~/go/src/speech_denoise_go
cp integerdata.gob ~/kitchen/voice_denoising_react/public
cd ~/go/src/speech_denoise_go
#go run main.go
