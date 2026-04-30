/**
 * UE5 Promo Site - 主要 JavaScript 交互代码
 * 功能：移动端菜单、滚动导航栏、平滑滚动、动画效果
 */

(function() {
    'use strict';

    // DOM 元素引用
    const header = document.querySelector('.header');
    const mobileMenuBtn = document.querySelector('.mobile-menu-btn');
    const navMenu = document.querySelector('.nav-links');
    const navLinks = document.querySelectorAll('.nav-link');
    const heroSection = document.querySelector('.hero');

    // 初始化
    document.addEventListener('DOMContentLoaded', function() {
        initMobileMenu();
        initScrollEffects();
        initSmoothScroll();
        initAnimations();
        console.log('UE5 Promo Site initialized successfully');
    });

    // 移动端菜单切换
    function initMobileMenu() {
        if (!mobileMenuBtn || !navMenu) return;

        mobileMenuBtn.addEventListener('click', function() {
            navMenu.classList.toggle('active');
            mobileMenuBtn.classList.toggle('active');
            
            // 切换 ARIA 属性
            const isExpanded = mobileMenuBtn.getAttribute('aria-expanded') === 'true';
            mobileMenuBtn.setAttribute('aria-expanded', !isExpanded);
        });

        // 点击导航链接后关闭菜单
        navLinks.forEach(function(link) {
            link.addEventListener('click', function() {
                navMenu.classList.remove('active');
                mobileMenuBtn.classList.remove('active');
                mobileMenuBtn.setAttribute('aria-expanded', 'false');
            });
        });

        // 点击页面其他区域关闭菜单
        document.addEventListener('click', function(event) {
            if (!navMenu.contains(event.target) && !mobileMenuBtn.contains(event.target)) {
                navMenu.classList.remove('active');
                mobileMenuBtn.classList.remove('active');
                mobileMenuBtn.setAttribute('aria-expanded', 'false');
            }
        });
    }

    // 滚动效果
    function initScrollEffects() {
        let lastScrollY = window.scrollY;
        let ticking = false;

        function updateScroll() {
            const currentScrollY = window.scrollY;

            // 导航栏背景效果
            if (currentScrollY > 50) {
                header.classList.add('scrolled');
            } else {
                header.classList.remove('scrolled');
            }

            // 滚动方向检测（可用于显示/隐藏导航栏）
            if (currentScrollY > lastScrollY && currentScrollY > 100) {
                header.classList.add('scroll-up');
            } else {
                header.classList.remove('scroll-up');
            }

            lastScrollY = currentScrollY;
            ticking = false;
        }

        window.addEventListener('scroll', function() {
            if (!ticking) {
                window.requestAnimationFrame(updateScroll);
                ticking = true;
            }
        }, { passive: true });

        // 初始检查
        updateScroll();
    }

    // 平滑滚动
    function initSmoothScroll() {
        document.querySelectorAll('a[href^="#"]').forEach(function(anchor) {
            anchor.addEventListener('click', function(e) {
                const targetId = this.getAttribute('href');
                if (targetId === '#') return;

                const targetElement = document.querySelector(targetId);
                if (targetElement) {
                    e.preventDefault();
                    
                    const headerHeight = header ? header.offsetHeight : 0;
                    const targetPosition = targetElement.offsetTop - headerHeight;

                    window.scrollTo({
                        top: targetPosition,
                        behavior: 'smooth'
                    });
                }
            });
        });
    }

    // 动画效果 - 使用 Intersection Observer
    function initAnimations() {
        // 特性卡片动画
        const featureCards = document.querySelectorAll('.feature-card');
        const techItems = document.querySelectorAll('.tech-item');
        const galleryItems = document.querySelectorAll('.gallery-item');
        const downloadCard = document.querySelector('.download-card');

        // 创建观察器
        const observerOptions = {
            root: null,
            rootMargin: '0px',
            threshold: 0.1
        };

        const observer = new IntersectionObserver(function(entries) {
            entries.forEach(function(entry) {
                if (entry.isIntersecting) {
                    entry.target.classList.add('visible');
                    observer.unobserve(entry.target);
                }
            });
        }, observerOptions);

        // 观察所有需要动画的元素
        featureCards.forEach(function(card, index) {
            card.style.transitionDelay = (index * 0.1) + 's';
            observer.observe(card);
        });

        techItems.forEach(function(item, index) {
            item.style.transitionDelay = (index * 0.1) + 's';
            observer.observe(item);
        });

        galleryItems.forEach(function(item, index) {
            item.style.transitionDelay = (index * 0.05) + 's';
            observer.observe(item);
        });

        if (downloadCard) {
            observer.observe(downloadCard);
        }

        // Hero 区域动画
        if (heroSection) {
            setTimeout(function() {
                heroSection.classList.add('loaded');
            }, 100);
        }
    }

    // 添加滚动进度条（可选功能）
    function initScrollProgress() {
        const progressBar = document.createElement('div');
        progressBar.className = 'scroll-progress';
        document.body.appendChild(progressBar);

        window.addEventListener('scroll', function() {
            const windowHeight = window.innerHeight;
            const documentHeight = document.documentElement.scrollHeight - windowHeight;
            const scrolled = (window.scrollY / documentHeight) * 100;
            progressBar.style.width = scrolled + '%';
        }, { passive: true });
    }

    // 图片懒加载（如果有图片元素）
    function initLazyLoading() {
        const images = document.querySelectorAll('img[data-src]');
        
        if ('IntersectionObserver' in window) {
            const imageObserver = new IntersectionObserver(function(entries) {
                entries.forEach(function(entry) {
                    if (entry.isIntersecting) {
                        const img = entry.target;
                        img.src = img.dataset.src;
                        img.removeAttribute('data-src');
                        imageObserver.unobserve(img);
                    }
                });
            });

            images.forEach(function(img) {
                imageObserver.observe(img);
            });
        } else {
            // 降级处理
            images.forEach(function(img) {
                img.src = img.dataset.src;
                img.removeAttribute('data-src');
            });
        }
    }

    // 按钮点击波纹效果
    function initRippleEffect() {
        const buttons = document.querySelectorAll('.btn');
        
        buttons.forEach(function(button) {
            button.addEventListener('click', function(e) {
                const rect = button.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const y = e.clientY - rect.top;

                const ripple = document.createElement('span');
                ripple.className = 'ripple';
                ripple.style.left = x + 'px';
                ripple.style.top = y + 'px';

                button.appendChild(ripple);

                setTimeout(function() {
                    ripple.remove();
                }, 600);
            });
        });
    }

    // 键盘导航支持
    function initKeyboardNavigation() {
        document.addEventListener('keydown', function(e) {
            // ESC 键关闭移动菜单
            if (e.key === 'Escape' && navMenu && navMenu.classList.contains('active')) {
                navMenu.classList.remove('active');
                mobileMenuBtn.classList.remove('active');
                mobileMenuBtn.setAttribute('aria-expanded', 'false');
                mobileMenuBtn.focus();
            }
        });
    }

    // 初始化所有功能
    initKeyboardNavigation();
    
})();
