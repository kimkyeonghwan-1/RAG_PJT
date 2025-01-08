/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx,html}"],
  theme: {
    extend: {
      colors: {
        customSky100: "#f8f9fb",
        customSky300: "#e1ecf7",
        customSky500: "#aecbeb",
        customSky700: "#83b0e1",
        customSky900: "#71a5de",
        customPink: "#efe9f4",
      },
    },
  },
  plugins: [],
};