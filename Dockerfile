# Minimal image used only to distribute EchoHands trained models
FROM alpine:3.20

# Store the trained models inside the image
COPY models /models