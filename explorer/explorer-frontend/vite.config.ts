import { defineConfig } from 'vite'
// @ts-ignore
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [
    tailwindcss(),
  ],
})