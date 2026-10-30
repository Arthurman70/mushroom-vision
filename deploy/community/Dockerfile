FROM node:24-slim
WORKDIR /app
COPY --chown=node:node community/ /app/
ENV NODE_ENV=production HOST=0.0.0.0 PORT=8791 BASE_PATH=/mushroom-lab/ DATA_DIR=/data COLLECTION_DIR=/collection
USER node
EXPOSE 8791
CMD ["node", "server.mjs"]
