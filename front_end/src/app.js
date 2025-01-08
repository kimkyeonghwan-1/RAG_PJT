// import "regenerator-runtime/runtime"; // if needed for async/await in older browsers

const chatContainer = document.getElementById("chat-container");
const messageForm = document.getElementById("message-form");
const userInput = document.getElementById("user-input");
const newChatBtn = document.getElementById("new-chat-btn");

const BASE_URL = process.env.API_ENDPOINT;

let db;

async function initDB() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open("myChatDB", 1);
    request.onupgradeneeded = function (e) {
      db = e.target.result;
      if (!db.objectStoreNames.contains("chats")) {
        db.createObjectStore("chats", { keyPath: "id", autoIncrement: true });
      }
      if (!db.objectStoreNames.contains("metadata")) {
        db.createObjectStore("metadata", { keyPath: "key" });
      }
    };
    request.onsuccess = function (e) {
      db = e.target.result;
      resolve();
    };
    request.onerror = function (e) {
      reject(e);
    };
  });
}

async function saveMessage(role, content) {
  return new Promise((resolve, reject) => {
    const tx = db.transaction("chats", "readwrite");
    const store = tx.objectStore("chats");
    store.add({ role, content });
    tx.oncomplete = () => resolve();
    tx.onerror = (e) => reject(e);
  });
}

async function getAllMessages() {
  return new Promise((resolve, reject) => {
    const tx = db.transaction("chats", "readonly");
    const store = tx.objectStore("chats");
    const req = store.getAll();
    req.onsuccess = () => resolve(req.result);
    req.onerror = (e) => reject(e);
  });
}

async function clearAllData() {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(["chats", "metadata"], "readwrite");
    tx.objectStore("chats").clear();
    tx.objectStore("metadata").clear();
    tx.oncomplete = () => resolve();
    tx.onerror = (e) => reject(e);
  });
}

// Create a message bubble
function createMessageBubble(content, sender = "user") {
  const wrapper = document.createElement("div");
  wrapper.classList.add("mb-6", "flex", "items-start", "space-x-3");


  // 개별 정렬 방향 설정
  if (sender === "user") {
    wrapper.classList.add("flex-row-reverse", "space-x-reverse", "space-x-3"); // Q는 오른쪽 정렬 // 수정!!
  }

  // Avatar
  // 공통 프로필 아이콘 디자인
  const avatar = document.createElement("div");
  avatar.classList.add(
    "w-10",
    "h-10",
    "rounded-full",
    "flex-shrink-0",
    "flex",
    "items-center",
    "justify-center",
    "font-bold",
    "text-white"
  );

  // 개별 프로필 아이콘 디자인
  if (sender === "assistant") {
    avatar.classList.add("bg-gradient-to-br", "from-indigo-300", "to-blue-600");
    avatar.textContent = "A";
  } else {
    avatar.classList.add("bg-gradient-to-br","from-customPink","to-customSky700");
    avatar.textContent = "Q";
  }

  // Bubble
  // 공통 말풍선 디자인
  const bubble = document.createElement("div");
  bubble.classList.add(
    "max-w-full",
    "md:max-w-2xl",
    "p-3",
    "rounded-lg",
    "whitespace-pre-wrap",
    "leading-relaxed",
    "shadow-sm"
  );

  // 개별 말풍선 디자인
  if (sender === "user") {
    bubble.classList.add("bg-customSky100", "text-gray-600");
  } else {
    bubble.classList.add("bg-gradient-to-br","from-customSky500","to-customSky700","text-white");
  }

  bubble.textContent = content;

  wrapper.appendChild(avatar);
  wrapper.appendChild(bubble);
  return wrapper;
}


// Scroll to bottom
function scrollToBottom() {
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

async function getChatResponse(userMessage) {
  const payload = { message: userMessage }; // 요청 데이터는 단순 문자열로 구성

  const response = await fetch(`${BASE_URL}/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      // CORS 관련 헤더 추가
      'Accept': 'application/json',
      'Access-Control-Allow-Origin': '*'
    },
    credentials: 'include',  // 필요한 경우 쿠키 포함
    body: JSON.stringify(payload), // JSON 형식으로 전송
  });

  if (!response.ok) {
    throw new Error("Network response was not ok");
  }

  const data = await response.json();
  return data.reply;
}

messageForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const message = userInput.value.trim();
  if (!message) return;

  chatContainer.appendChild(createMessageBubble(message, "user"));
  await saveMessage("user", message);

  userInput.value = "";
  scrollToBottom();

  try {
    const response = await getChatResponse(message);
    chatContainer.appendChild(createMessageBubble(response, "assistant"));
    await saveMessage("assistant", response);
    scrollToBottom();
  } catch (error) {
    console.error("Error fetching response:", error);
    const errMsg = "Error fetching response. Check console.";
    chatContainer.appendChild(createMessageBubble(errMsg, "assistant"));
    await saveMessage("assistant", errMsg);
    scrollToBottom();
  }
});

async function loadExistingMessages() {
  const allMsgs = await getAllMessages();
  for (const msg of allMsgs) {
    chatContainer.appendChild(createMessageBubble(msg.content, msg.role));
  }
  scrollToBottom();
}

newChatBtn.addEventListener("click", async () => {
  // Clear DB data and UI
  await clearAllData();
  chatContainer.innerHTML = "";
  // Now user can start a new chat fresh
});

initDB().then(loadExistingMessages);