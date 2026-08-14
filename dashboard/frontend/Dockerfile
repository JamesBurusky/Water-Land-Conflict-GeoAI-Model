# Stage 1: build the React app
FROM node:20-slim AS build
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
# VITE_API_BASE_URL is baked in at BUILD time (Vite env vars are
# compile-time, not runtime) -- set via docker-compose build args.
ARG VITE_API_BASE_URL=/api
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL
RUN npm run build

# Stage 2: serve the built static files with nginx, proxying /api to
# the backend container -- this is the ONLY container that needs a
# port exposed to the host, so the whole dashboard is reachable at a
# single IP/port on Contabo.
FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
