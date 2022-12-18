# パート 1 - インポート
import webbrowser
import PySimpleGUI as sg

# リンク用辞書作成
dict_link = {
    'Google':'https://www.google.com/',
    'Youtube':'https://www.youtube.com/',
    'Amazon':'https://www.amazon.co.jp/',
}

layout_group1 = []
layout_group1_text = [sg.Text('サイトリンク', font=('メイリオ', 10), auto_size_text=True)]
layout_group1_Buttons = [sg.Button(name) for name in dict_link.keys()]
layout_group1.append(layout_group1_text)
layout_group1.append(layout_group1_Buttons)

layout_group2 = []
layout_group2_text = [sg.Text('リンクをすべて開く', font=('メイリオ', 10), auto_size_text=True)]
layout_group2_Buttons = [sg.Button('Open All')]
layout_group2.append(layout_group2_text)
layout_group2.append(layout_group2_Buttons)

layout_group3 = []
layout_group3_text = [sg.Text('終了ボタン', font=('メイリオ', 10), auto_size_text=True)]
layout_group3_Buttons = [sg.Button('Quit')]
layout_group3.append(layout_group2_text)
layout_group3.append(layout_group2_Buttons)


# ウィンドウの内容を定義する
# パート 2 - レイアウト
layout = [ layout_group1, layout_group2 ]

# ウィンドウを作成する
# パート 3- ウィンドウ定義
window = sg.Window('リンク集', layout, no_titlebar=False, alpha_channel=1.0)
                                                
# ウィンドウを表示し、対話する
# パート 4- イベントループまたは Window.read 呼び出し
# event, values = window.read()
while True:
    # イベント読み取り
    event, _ = window.read()
    # layout_group1 の処理
    for name, url in dict_link.items():
        if event == name:
            webbrowser.open(url)
    # layout_group2 の処理
    if event == 'Open All':
        for url in dict_link.values():
            webbrowser.open(url)
    if event == None:  # 終了条件（ None: クローズボタンを押した場合）
        break

# # 収集された情報で何かをする
# print('ハロー ', values[0], "PySimpleGUIを試してくれてありがとう!")

# # 画面から削除して終了
# # パート 5 - ウィンドウを閉じる
# window.close()