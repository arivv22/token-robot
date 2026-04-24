package main

import (
	"bytes"
	"encoding/json"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"regexp"
	"strings"

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

	// Check for text message with token
	text, hasText := message["text"].(string)
	
	// Check for photo message
	_, hasPhoto := message["photo"].([]interface{})
	
	if !hasText && !hasPhoto {
		sendMessage(chatID, "Kirim token dengan format: token :22592997522702236675")
		return
	}

	var token string
	
	if hasText {
		// Parse token from text message
		token = extractTokenFromText(text)
		if len(token) != 20 {
			sendMessage(chatID, "Token tidak valid. Gunakan format: token :22592997522702236675")
			return
		}
	} else {
		// Handle photo (legacy support)
		photoArr := message["photo"].([]interface{})
		lastPhoto := photoArr[len(photoArr)-1].(map[string]interface{})
		fileID := lastPhoto["file_id"].(string)

		fileURL := getFileURL(fileID)
		imgBytes := downloadFile(fileURL)

		token = callOCR(imgBytes)

		if len(token) != 20 {
			sendMessage(chatID, "Token tidak valid, coba ulang.")
			return
		}
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

func extractTokenFromText(text string) string {
	// Convert to lowercase for case-insensitive matching
	lowerText := strings.ToLower(text)
	
	// Token patterns similar to Python version
	tokenPatterns := []string{
		`2259\s*2997\s*5227\s*0223\.?6675`,           // Specific pattern with optional dot
		`2259\s*\d{4}\s*\d{4}\s*\d{4}\.?\d{4}`,       // Pattern starting with 2259
		`(\d{4}[\s\.]*\d{4}[\s\.]*\d{4}[\s\.]*\d{4}[\s\.]*\d{4})`, // General 5-group pattern
	}
	
	// Try each pattern
	for _, pattern := range tokenPatterns {
		tokenRegex := regexp.MustCompile(pattern)
		matches := tokenRegex.FindStringSubmatch(text)
		
		if len(matches) >= 1 {
			// Extract all digits from the match
			digitRegex := regexp.MustCompile(`\d`)
			digitMatches := digitRegex.FindAllString(matches[0], -1)
			token := strings.Join(digitMatches, "")
			
			if len(token) >= 20 {
				return token[:20]
			}
		}
	}
	
	// Fallback: look for any sequence of digits and spaces (10+ characters)
	groupRegex := regexp.MustCompile(`[\d\s]{10,}`)
	groupMatches := groupRegex.FindAllString(text, -1)
	
	bestToken := ""
	
	for _, group := range groupMatches {
		digitRegex := regexp.MustCompile(`\d`)
		digitMatches := digitRegex.FindAllString(group, -1)
		joined := strings.Join(digitMatches, "")
		
		if len(joined) > len(bestToken) {
			bestToken = joined
		}
	}
	
	if len(bestToken) >= 20 {
		return bestToken[:20]
	}
	
	// Final fallback: look for any 20-digit sequence
	digitRegex := regexp.MustCompile(`\d{20}`)
	matches := digitRegex.FindStringSubmatch(text)
	
	if len(matches) >= 1 {
		return matches[0]
	}
	
	return ""
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