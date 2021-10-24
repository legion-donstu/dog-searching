const element = (tag, classes = [], content) => {
    const node = document.createElement(tag)

    if (classes.length) {
        node.classList.add(...classes)
    }

    if (content) {
        node.textContent = content
    }

    return node
}

function noop() {}

function upload(selector, options = {}) {
    let files = []
    const onUpload = options.onUpload ?? noop
    const  input = document.querySelector(selector)
    const preview = element('div', ['preview'])
    const open = element('button', ['btn'], 'Открыть')
    const upload = element('button', ['btn', 'primary'], 'Загрузить')
    upload.style.display = 'none'

    if (options.multi) {
        input.setAttribute('multiple', true)
    }

    if (options.accept && Array.isArray(options.accept)) {
        input.setAttribute('accept', options.accept.join(','))
    }

    input.insertAdjacentElement('afterend', preview)
    input.insertAdjacentElement('afterend', upload)
    input.insertAdjacentElement('afterend', open)

    const triggerInput = () => input.click()
      
    const changeHandler = event => {
      if (!event.target.files.length) {
          return
      }
      files = Array.from(event.target.files)
      preview.innerHTML = ''
      upload.style.display = 'inline'

        files.forEach(file => {
            if (!file.type.match('image')) {
                return
            }

            const reader = new FileReader()

            reader.onload = ev => {
                const src = ev.target.result
                preview.insertAdjacentHTML('afterbegin', `
                    <div class="preview-image">
                        <img src="${src}" alt="${file.name}" />
                    </div>
                `)
            }

            reader.readAsDataURL(file)
        })
        


    }

    const removeHandler = event => {
      if (!event.target.dataset.name) {
          return
      }
    }

    /*const {name} = event.target.dataset
    files = files.filter(file => file.name !== name)
      */
    if (!files.length) {
        upload.style.display = 'none'
    }

    const clearPreview = el => {
        el.style.bottom = '4px'
        el.innerHTML = '<div class="preview-info-progress"><div'
    }

    const uploadHandler = () => {
        onUpload(files)

    }

    open.addEventListener('click', triggerInput)
    input.addEventListener('change', changeHandler)
    upload.addEventListener('click', uploadHandler)
}
