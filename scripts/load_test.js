import http from "k6/http";
import { check, sleep } from "k6";
import { Rate, Trend } from "k6/metrics";

const API_KEY = "admin_bismillah";
const BASE_URL = __ENV.BASE_URL || "http://localhost:3001";

const recommendLatency = new Trend("recommend_latency_ms");
const recommendErrors = new Rate("recommend_errors");

export const options = {
  stages: [
    { duration: "10s", target: 5 },
    { duration: "20s", target: 20 },
    { duration: "10s", target: 50 },
    { duration: "20s", target: 50 },
    { duration: "10s", target: 0 },
  ],
  thresholds: {
    recommend_latency_ms: ["p(95)<50", "p(99)<100"],
    recommend_errors: ["rate<0.01"],
    http_req_duration: ["p(95)<200"],
  },
};

function randomUserId() {
  const max = __ENV.MAX_USERS ? parseInt(__ENV.MAX_USERS) : 10000;
  const min = __ENV.MIN_USERS ? parseInt(__ENV.MIN_USERS) : 0;
  return Math.floor(Math.random() * (max - min)) + min;
}

export default function () {
  const userId = randomUserId();

  const payload = JSON.stringify({ user_id: userId });
  const params = {
    headers: {
      "Content-Type": "application/json",
      "x-api-key": API_KEY,
    },
  };

  const res = http.post(`${BASE_URL}/recommend`, payload, params);

  check(res, {
    "status is 200": (r) => r.status === 200,
    "has ranked items": (r) => {
      const body = r.json();
      return body.ranked && body.ranked.length > 0;
    },
    "has latency_ms": (r) => {
      const body = r.json();
      return body.latency_ms !== undefined;
    },
  });

  if (res.status === 200) {
    const body = res.json();
    recommendLatency.add(body.latency_ms);
  } else {
    recommendErrors.add(1);
  }

  sleep(0.1);
}
