from PyARTOS.GUI.MainFrame import MainFrame

app = MainFrame()
app.mainloop()
try:
    app.destroy()
except:
    pass
