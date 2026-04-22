package main

import (
	"bytes"
	"encoding/json"
	"io"
	"mime/multipart"
	"net/http"
	"os"

	"github.com/gin-gonic/gin"
)

var TELEGRAM_TOKEN = "YOUR_BOT_TOKEN"
var ESP32_URL = "http://192.168.1.100/input-token"

func main() {
	r := gin.Default()
	r.POST("/webhook", handleTelegram)
	r.Run(":8080")
}

func handleTelegram(c *gin.Context) {
	var body map[string]interface{}
	if err := c.BindJSON(&body); err != nil {
		return
	}

	message := body["message"].(map[string]interface{})
	chat := message["chat"].(map[string]interface{})
	chatID := int(chat["id"].(float64))

	photoArr, ok := message["photo"].([]interface{})
	if !ok {
		sendMessage(chatID, "Kirim gambar token ya.")
		return
	}

	// ambil gambar resolusi tertinggi
	lastPhoto := photoArr[len(photoArr)-1].(map[string]interface{})
	fileID := lastPhoto["file_id"].(string)

	fileURL := getFileURL(fileID)
	imgBytes := downloadFile(fileURL)

	token := callOCR(imgBytes)

	if len(token) != 20 {
		sendMessage(chatID, "Token tidak valid, coba ulang.")
		return
	}

	sendMessage(chatID, "Token terbaca: "+token)

	sendToESP32(token)

	sendMessage(chatID, "Token sedang diinput robot 🤖")
}

func getFileURL(fileID string) string {
	resp, _ := http.Get("https://api.telegram.org/bot" + TELEGRAM_TOKEN + "/getFile?file_id=" + fileID)
	body, _ := io.ReadAll(resp.Body)

	var result map[string]interface{}
	json.Unmarshal(body, &result)

	filePath := result["result"].(map[string]interface{})["file_path"].(string)
	return "https://api.telegram.org/file/bot" + TELEGRAM_TOKEN + "/" + filePath
}

func downloadFile(url string) []byte {
	resp, _ := http.Get(url)
	data, _ := io.ReadAll(resp.Body)
	return data
}

func callOCR(image []byte) string {
	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)

	part, _ := writer.CreateFormFile("image", "img.jpg")
	part.Write(image)
	writer.Close()

	req, _ := http.NewRequest("POST", "http://localhost:5000/ocr", body)
	req.Header.Set("Content-Type", writer.FormDataContentType())

	client := &http.Client{}
	resp, _ := client.Do(req)
	respBody, _ := io.ReadAll(resp.Body)

	var result map[string]interface{}
	json.Unmarshal(respBody, &result)

	return result["token"].(string)
}

func sendToESP32(token string) {
	jsonData, _ := json.Marshal(map[string]string{
		"token": token,
	})

	http.Post(ESP32_URL, "application/json", bytes.NewBuffer(jsonData))
}

func sendMessage(chatID int, text string) {
	url := "https://api.telegram.org/bot" + TELEGRAM_TOKEN + "/sendMessage"

	payload := map[string]interface{}{
		"chat_id": chatID,
		"text":    text,
	}

	jsonData, _ := json.Marshal(payload)
	http.Post(url, "application/json", bytes.NewBuffer(jsonData))
}