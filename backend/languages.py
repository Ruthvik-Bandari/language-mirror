"""Multi-Language Support for Language Mirror Pro"""
import random

SUPPORTED_LANGUAGES = {
    "italian": {
        "name": "Italian",
        "dialects": ["standard", "sicilian", "roman", "milanese", "neapolitan"],
        "responses": {
            "greeting": [
                ("Ciao! Come stai oggi? Sono qui per aiutarti a imparare l'italiano.", "Hello! How are you today? I'm here to help you learn Italian."),
                ("Buongiorno! Pronto per la nostra lezione?", "Good morning! Ready for our lesson?"),
            ],
            "encouragement": [
                ("Ottimo lavoro! Stai facendo progressi fantastici!", "Great job! You're making fantastic progress!"),
                ("Benissimo! La tua pronuncia sta migliorando molto!", "Very good! Your pronunciation is improving a lot!"),
                ("Eccellente! Sei sulla strada giusta!", "Excellent! You're on the right track!"),
            ],
            "question": [
                ("Che cosa ti piace fare nel tempo libero?", "What do you like to do in your free time?"),
                ("Raccontami della tua giornata.", "Tell me about your day."),
                ("Qual è il tuo cibo italiano preferito?", "What's your favorite Italian food?"),
            ],
            "fallback": [
                ("Interessante! Continua così!", "Interesting! Keep going!"),
                ("Capisco! Dimmi di più.", "I understand! Tell me more."),
            ],
        },
        "suggestions": ["Grazie mille!", "Come si dice...?", "Può ripetere?", "Non capisco", "Perfetto!"]
    },
    "japanese": {
        "name": "Japanese",
        "dialects": ["standard", "osaka", "kyoto", "hokkaido"],
        "responses": {
            "greeting": [
                ("こんにちは！今日も日本語を勉強しましょう！", "Hello! Let's study Japanese today too!"),
                ("ようこそ！日本語の練習を始めましょう！", "Welcome! Let's start practicing Japanese!"),
            ],
            "encouragement": [
                ("すごいですね！上手になっています！", "That's amazing! You're getting better!"),
                ("よくできました！その調子で頑張ってください！", "Well done! Keep up the good work!"),
                ("素晴らしい！発音がとても良くなりましたね！", "Wonderful! Your pronunciation has improved!"),
            ],
            "question": [
                ("趣味は何ですか？", "What are your hobbies?"),
                ("日本に行ったことがありますか？", "Have you ever been to Japan?"),
                ("好きな日本の食べ物は何ですか？", "What's your favorite Japanese food?"),
            ],
            "fallback": [
                ("面白いですね！続けてください！", "That's interesting! Please continue!"),
                ("なるほど！もっと教えてください。", "I see! Tell me more."),
            ],
        },
        "suggestions": ["ありがとうございます", "もう一度お願いします", "わかりません", "すみません"]
    },
    "spanish": {
        "name": "Spanish",
        "dialects": ["castilian", "mexican", "argentinian", "colombian"],
        "responses": {
            "greeting": [
                ("¡Hola! ¿Cómo estás hoy? ¡Vamos a practicar español!", "Hello! How are you today? Let's practice Spanish!"),
                ("¡Bienvenido! ¡Empecemos nuestra lección de español!", "Welcome! Let's start our Spanish lesson!"),
            ],
            "encouragement": [
                ("¡Muy bien! ¡Estás progresando mucho!", "Very good! You're progressing a lot!"),
                ("¡Excelente trabajo! Tu español está mejorando.", "Excellent work! Your Spanish is improving."),
                ("¡Fantástico! ¡Sigue así!", "Fantastic! Keep it up!"),
            ],
            "question": [
                ("¿Qué te gusta hacer en tu tiempo libre?", "What do you like to do in your free time?"),
                ("¿Has visitado algún país hispanohablante?", "Have you visited any Spanish-speaking country?"),
                ("¿Cuál es tu comida favorita?", "What's your favorite food?"),
            ],
            "fallback": [
                ("¡Interesante! ¡Continúa!", "Interesting! Continue!"),
                ("¡Entiendo! Cuéntame más.", "I understand! Tell me more."),
            ],
        },
        "suggestions": ["¡Gracias!", "¿Puede repetir?", "No entiendo", "¿Cómo se dice...?"]
    },
    "french": {
        "name": "French",
        "dialects": ["parisian", "quebec", "belgian", "swiss"],
        "responses": {
            "greeting": [
                ("Bonjour! Comment allez-vous? Pratiquons le français ensemble!", "Hello! How are you? Let's practice French together!"),
                ("Bienvenue! Commençons notre leçon de français!", "Welcome! Let's start our French lesson!"),
            ],
            "encouragement": [
                ("Très bien! Vous faites de grands progrès!", "Very good! You're making great progress!"),
                ("Excellent travail! Votre français s'améliore!", "Excellent work! Your French is improving!"),
                ("Magnifique! Continuez comme ça!", "Magnificent! Keep it up!"),
            ],
            "question": [
                ("Qu'est-ce que vous aimez faire pendant votre temps libre?", "What do you like to do in your free time?"),
                ("Avez-vous déjà visité la France?", "Have you ever visited France?"),
                ("Quelle est votre nourriture préférée?", "What's your favorite food?"),
            ],
            "fallback": [
                ("Intéressant! Continuez!", "Interesting! Continue!"),
                ("Je comprends! Dites-m'en plus.", "I understand! Tell me more."),
            ],
        },
        "suggestions": ["Merci beaucoup!", "Pouvez-vous répéter?", "Je ne comprends pas", "Comment dit-on...?"]
    },
    "german": {
        "name": "German",
        "dialects": ["standard", "bavarian", "austrian", "swiss"],
        "responses": {
            "greeting": [
                ("Hallo! Wie geht es Ihnen? Lass uns Deutsch üben!", "Hello! How are you? Let's practice German!"),
                ("Willkommen! Beginnen wir mit unserer Deutschstunde!", "Welcome! Let's start our German lesson!"),
            ],
            "encouragement": [
                ("Sehr gut! Sie machen große Fortschritte!", "Very good! You're making great progress!"),
                ("Ausgezeichnet! Ihr Deutsch wird besser!", "Excellent! Your German is getting better!"),
                ("Wunderbar! Weiter so!", "Wonderful! Keep it up!"),
            ],
            "question": [
                ("Was machen Sie gerne in Ihrer Freizeit?", "What do you like to do in your free time?"),
                ("Waren Sie schon einmal in Deutschland?", "Have you ever been to Germany?"),
                ("Was ist Ihr Lieblingsessen?", "What's your favorite food?"),
            ],
            "fallback": [
                ("Interessant! Machen Sie weiter!", "Interesting! Keep going!"),
                ("Ich verstehe! Erzählen Sie mehr.", "I understand! Tell me more."),
            ],
        },
        "suggestions": ["Danke schön!", "Können Sie das wiederholen?", "Ich verstehe nicht", "Wie sagt man...?"]
    },
}
