# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 16:16:44 2019

@author: Chacrew
"""

from urllib.request import Request, urlopen

## https://core.telegram.org/bots/api#getupdates

API_KEY = '936714777:AAGFCBbeOAClrTsgmMMOsYG3HkaV7Ck5p-w'
WEBHOOK_URL = 'http://9c345afe.ngrok.io'
BOT_INFO_URL = 'https://api.telegram.org/bot{API_KEY}/getMe'.format(API_KEY=API_KEY)
BOT_UPDATE_URL = 'https://api.telegram.org/bot{API_KEY}/getUpdates'.format(API_KEY=API_KEY)
BOT_SET_WEBHOOK_URL = 'https://api.telegram.org/bot{API_KEY}/setWebhook?url={WEBHOOK_URL}'\
    .format(API_KEY=API_KEY, WEBHOOK_URL=WEBHOOK_URL)
BOT_WEBHOOK_INFO_URL = 'https://api.telegram.org/bot{API_KEY}/getWebhookInfo'.format(API_KEY=API_KEY)
BOT_DELETE_URL = 'https://api.telegram.org/bot{API_KEY}/deleteWebhook'.format(API_KEY=API_KEY)
BOT_DELETE_MSG_URL = 'https://api.telegram.org/bot{API_KEY}/deleteMessage?chat_id={chat_id}&message_id={msg_id}'\
    .format(API_KEY=API_KEY, chat_id=727895200, msg_id=514)


def bot_info_call():
    """
    bot 의 정보를 출력
    """
    request = Request(BOT_INFO_URL)
    request.get_method = lambda: 'GET'
    response_body = urlopen(request).read().decode('utf-8')
    print(response_body)


def bot_update_call():
    """
    bot 의 업데이트 정보
    """
    request = Request(BOT_UPDATE_URL)
    request.get_method = lambda: 'GET'
    response_body = urlopen(request).read().decode('utf-8')
    print(response_body)


def bot_set_webhook_call():
    """
    bot 과 webhook 을 연결
    """
    request = Request(BOT_SET_WEBHOOK_URL)
    request.get_method = lambda: 'GET'
    response_body = urlopen(request).read().decode('utf-8')
    print(response_body)


def delete_webhook():
    """
    bot 의 Webhook 을 제거
    """
    request = Request(BOT_DELETE_URL)
    request.get_method = lambda: 'GET'
    response_body = urlopen(request).read().decode('utf-8')
    print(response_body)


def delete_msg_call():
    """
    Telegram Server에 남아있는 Msg 삭제

    """
    request = Request(BOT_DELETE_MSG_URL)
    request.get_method = lambda: 'GET'
    response_body = urlopen(request).read().decode('utf-8')
    print(response_body)


def web_hook_info():
    """
    Webhook 에 대한 정보 출력
    """
    request = Request(BOT_WEBHOOK_INFO_URL)
    request.get_method = lambda: 'GET'
    response_body = urlopen(request).read().decode('utf-8')
    print(response_body)

bot_set_webhook_call()