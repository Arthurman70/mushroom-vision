FROM node:24-alpine
WORKDIR /app
COPY package.json server.mjs ./
COPY core ./core
COPY flight ./flight
COPY config ./config
COPY public ./public
RUN mkdir -p /app/data && chown -R node:node /app
USER node
ENV MYCOSCOUT_HOST=0.0.0.0 MYCOSCOUT_PORT=8787
EXPOSE 8787
CMD ["node", "server.mjs"]
