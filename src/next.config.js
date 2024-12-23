/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "standalone",
  async headers() {
    return [
      {
        source: "/(.*)", // 全ページに対して適用
        headers: [
          {
            key: "X-Frame-Options",
            value: "ALLOWALL" // iFrameでの表示を許可
          },
          {
            key: "Content-Security-Policy",
            value: "frame-ancestors 'self' *" // 任意のドメインでのiFrame埋め込みを許可
          },
          {
            key: "Access-Control-Allow-Origin",
            value: "*" // 任意のオリジンからのアクセスを許可
          },
        ],
      },
    ];
  },
};


module.exports = nextConfig;