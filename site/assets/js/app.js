upload('#file', {
    multi: true,
    accept: ['.png', '.jpg', 'jpeg'],
    onUpload(files) {
        console.log('files:', files)
    }
})
